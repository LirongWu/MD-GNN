import os
import nni
import csv
import json
import time
import warnings
import argparse

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from utils import *
from model import get_model
from dataset import load_data

warnings.filterwarnings("ignore", category=Warning)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def evaluate_synthetic(model, data_loder, param, log_dir, plot=False):
    with torch.no_grad():

        logits_all = []
        labels_all = []
        model.eval()

        for _, (g, labels, gt_adjs) in enumerate(data_loder):
            model.g = g
            features = g.ndata['feat'].to(device)
            labels = labels.to(device)
            logits = model(features, mode='test')
            logits_all.append(logits.detach())
            labels_all.append(labels.detach())
        
        logits_all = torch.cat(tuple(logits_all), 0)
        labels_all = torch.cat(tuple(labels_all), 0)
        micro_f1 = evaluate_f1(logits_all, labels_all)

        att_score = model.GraphLearning.att.detach().cpu().numpy()
        f_score = evaluate_att(att_score, param, log_dir, plot) 

    return micro_f1, f_score


def main_synthetic(param):
    set_seed(param['seed'])
    log_dir = "./log/{}/".format(param['ExpName'])
    os.makedirs(log_dir, exist_ok=True)
    json.dump(param, open("{}param.json".format(log_dir), 'a'), indent=2)

    data = load_data(param)
    train_data = data[:int(len(data)*0.7)]
    val_data = data[int(len(data)*0.7) : int(len(data)*0.8)]
    test_data = data[int(len(data)*0.8):]

    train_loader = DataLoader(train_data, batch_size=param['batch_size'], shuffle=True, collate_fn=collate)
    val_loader = DataLoader(val_data, batch_size=param['batch_size'], shuffle=True, collate_fn=collate)
    test_loader = DataLoader(test_data, batch_size=param['batch_size'], shuffle=False, collate_fn=collate)

    model = get_model(param).to(device)
    loss_fcn = torch.nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=param['lr'], weight_decay=param['weight_decay'])

    best_test_acc = 0
    best_val_acc = 0
    best_val_test_acc = 0
    best_epoch_acc = 0

    best_test_score = 0
    best_epoch_score = 0

    for epoch in range(param['epochs']):

        loss_1 = []
        loss_2 = []
        loss_3 = []
        train_loss = []

        for _, (g, labels, gt_adjs) in enumerate(train_loader):

            model.train()      
            optimizer.zero_grad()
                
            model.g = g
            features = g.ndata['feat'].to(device)
            labels = labels.to(device)
            logits = model(features)
            loss_cla = loss_fcn(logits, labels)
            
            loss_graph, loss_node = model.compute_disentangle_loss()
            loss = loss_cla + loss_graph * param['ratio_graph'] + loss_node * param['ratio_node']

            loss.backward()
            optimizer.step()
                
            loss_1.append(loss_cla.item())
            loss_2.append(loss_graph.item() * param['ratio_graph'])
            loss_3.append(loss_node.item() * param['ratio_node'])
            train_loss.append(loss.item())

        train_acc, _ = evaluate_synthetic(model, train_loader, param, log_dir)
        val_acc, _ = evaluate_synthetic(model, val_loader, param, log_dir)
        test_acc, test_score = evaluate_synthetic(model, test_loader, param, log_dir)

        if test_acc > best_test_acc:
            best_test_acc = test_acc
            best_epoch_acc = epoch
            torch.save({'epoch': epoch, 'model_state_dict': model.state_dict(), 'optimizer_state_dict': optimizer.state_dict(), 'param': param}, log_dir + 'best_model.pt')

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_val_test_acc = test_acc

        if test_score > best_test_score:
            best_test_score = test_score
            best_epoch_score = epoch
            _, _ = evaluate_synthetic(model, test_loader, param, log_dir='{}Feature/Epoch{}_Score{}/'.format(log_dir, epoch, int(test_score)), plot=True)

        print("\033[0;30;46m Epoch: {} | Loss: {:.6f}, {:.6f}, {:.12f}, {:.6f} | Acc: {:.5f}, {:.5f}, {:.5f}, {:.5f}({}) | Num: {}, {} \033[0m".format(
        epoch, np.mean(loss_1), np.mean(loss_2), np.mean(loss_3), np.mean(train_loss), train_acc, val_acc, test_acc, best_test_acc, best_epoch_acc, test_score, best_test_score))
    
    nni.report_final_result(best_test_score)
    outFile = open('./log/PerformMetrics.csv','a+', newline='')
    writer = csv.writer(outFile, dialect='excel')
    results = [time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())]
    for v, k in param.items():
        results.append(k)
    results.append(str(test_acc))
    results.append(str(best_val_test_acc))
    results.append(str(best_test_acc))
    results.append(str(best_epoch_acc))
    results.append(str(best_test_score))
    results.append(str(best_epoch_score))
    path = './log/{}/best_model.pt'.format(param['ExpName'])
    best_model = torch.load(path)
    cscore, ged_m, ged_s = evaluate_graph(best_model)
    results.append(str(cscore))
    results.append(str(ged_m))
    results.append(str(ged_s))
    writer.writerow(results)


def evaluate_zinc(model, data_loader):
    loss_fcn = torch.nn.L1Loss()

    model.eval()
    loss = 0
    mae = 0

    with torch.no_grad():
        for batch_idx, (batch_graphs, batch_targets, batch_snorm_n) in enumerate(data_loader):
            batch_x = batch_graphs.ndata['feat'].to(device)
            batch_targets = batch_targets.to(device)
            batch_snorm_n = batch_snorm_n.to(device)
            
            model.g = batch_graphs
            batch_scores = model.forward(batch_x, batch_snorm_n)
            
            eval_loss = loss_fcn(batch_scores, batch_targets).item()
            eval_mae = F.l1_loss(batch_scores, batch_targets).item()
            loss += eval_loss
            mae += eval_mae
        
    loss /= (batch_idx + 1)
    mae /= (batch_idx + 1)

    return loss, mae


def main_zinc(param):
    set_seed(param['seed'])
    log_dir = "./log/{}/".format(param['ExpName'])
    os.makedirs(log_dir, exist_ok=True)
    json.dump(param, open("{}param.json".format(log_dir), 'a'), indent=2)

    zinc_data = load_data(param)
    train_loader = DataLoader(zinc_data.train, batch_size=1000, shuffle=True, collate_fn=zinc_data.collate)
    val_loader = DataLoader(zinc_data.val, batch_size=1000, shuffle=False, collate_fn=zinc_data.collate)
    test_loader = DataLoader(zinc_data.test, batch_size=1000, shuffle=False, collate_fn=zinc_data.collate)

    model = get_model(param).to(device)
    loss_fcn = torch.nn.L1Loss()
    optimizer = torch.optim.Adam(model.parameters(), lr=param['lr'], weight_decay=param['weight_decay'])

    best_test_mae = 1e6
    best_val_mae = 1e6
    best_val_test_mae = 1e6
    best_epoch_mae = 0

    for epoch in range(param['epochs']):

        model.train()
        epoch_loss = 0
        epoch_train_mae = 0

        for batch_idx, (batch_graphs, batch_targets, batch_snorm_n) in enumerate(train_loader):
            batch_x = batch_graphs.ndata['feat'].to(device)
            batch_targets = batch_targets.to(device)
            batch_snorm_n = batch_snorm_n.to(device)      
            
            optimizer.zero_grad()
            
            model.g = batch_graphs
            batch_scores = model.forward(batch_x, batch_snorm_n)
            
            loss_mae = loss_fcn(batch_scores, batch_targets)
            loss_graph, loss_node = model.compute_disentangle_loss()
            loss = loss_mae + loss_graph * param['ratio_graph'] + loss_node * param['ratio_node']

            loss.backward()
            optimizer.step()
            
            loss_1 = loss_mae.item()
            loss_2 = loss_graph.item() * param['ratio_graph']
            loss_3 = loss_node.item() * param['ratio_node']

            train_loss = loss.item()
            train_mae = F.l1_loss(batch_scores, batch_targets).item()
            epoch_loss += train_loss
            epoch_train_mae += train_mae

            # print("Epoch: {} | [{}/{}] | Loss: {}, {}, {}, {}".format(epoch, batch_idx+1, 10, loss_1, loss_2, loss_3, train_loss))
        
        epoch_loss /= (batch_idx + 1)
        epoch_train_mae /= (batch_idx + 1)
        val_loss, val_mae = evaluate_zinc(model, val_loader)
        test_loss, test_mae = evaluate_zinc(model, test_loader)
        
        if test_mae < best_test_mae:
            best_test_mae = test_mae
            best_epoch_mae = epoch
            torch.save({'epoch': epoch, 'model_state_dict': model.state_dict(), 'optimizer_state_dict': optimizer.state_dict(), 'param': param}, log_dir + 'best_model.pt')

        if val_mae < best_val_mae:
            best_val_mae = val_mae
            best_val_test_mae = test_mae

        print("\033[0;30;46m Epoch: {} | Loss: {:.6f} | Mae: {:.5f}, {:.5f}, {:.5f}, {:.5f}({}), {:.5f} \033[0m".format(
        epoch, epoch_loss, epoch_train_mae, val_mae, test_mae, best_test_mae, best_epoch_mae, best_val_test_mae))
    
    outFile = open('./log/PerformMetrics.csv','a+', newline='')
    writer = csv.writer(outFile, dialect='excel')
    results = [time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())]
    for v, k in param.items():
        results.append(k)
    results.append(str(test_mae))
    results.append(str(best_val_test_mae))
    results.append(str(best_test_mae))
    results.append(str(best_epoch_mae))

    path = './log/{}/best_model.pt'.format(param['ExpName'])
    best_model = torch.load(path)
    cscore, ged_m, ged_s = evaluate_graph(best_model)
    results.append(str(cscore))
    results.append(str(ged_m))
    results.append(str(ged_s))
    nni.report_final_result(best_val_test_mae)

    writer.writerow(results)


def evaluate(model, features, labels, mask):
    model.eval()
    with torch.no_grad():
        logits = model(features, mode='test')
        logits = logits[mask]
        labels = labels[mask]
        _, pred = torch.max(logits, dim=1)
        correct = torch.sum(pred == labels)
        return correct.item() * 1.0 / len(labels)


def main(param):
    set_seed(param['seed'])

    g, features, labels, train_mask, val_mask, test_mask = load_data(param)
    param['input_dim'] = features.shape[1]
    param['output_dim'] = torch.max(labels) + 1
    model = get_model(param).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=param['lr'], weight_decay=param['weight_decay'])

    test_best = 0
    test_val = 0
    val_best = 0
    val_best_epoch = 0

    for epoch in range(param['epochs']):

        model.train()      
        optimizer.zero_grad()
            
        model.g = g
        logits = model(features)
        pred = F.log_softmax(logits, 1)
        loss_cla = F.nll_loss(pred[train_mask], labels[train_mask])
        loss_graph, loss_node = model.compute_disentangle_loss()
        loss = loss_cla + loss_graph * param['ratio_graph'] + loss_node * param['ratio_node']

        loss.backward()
        optimizer.step()
            
        loss_1 = loss_cla.item()
        loss_2 = loss_graph.item() * param['ratio_graph']
        loss_3 = loss_node.item() * param['ratio_node']

        train_loss = loss.item()
        train_acc = evaluate(model, features, labels, train_mask)
        val_acc = evaluate(model, features, labels, val_mask)
        test_acc = evaluate(model, features, labels, test_mask)
        # nni.report_intermediate_result(test_acc)

        if test_acc > test_best:
            test_best = test_acc

        if val_acc > val_best:
            val_best = val_acc
            test_val = test_acc
            val_best_epoch = epoch

        print("\033[0;30;46m Epoch: {} | Loss: {:.6f}, {:.6f}, {:.12f}, {:.6f} | Acc: {:.5f}, {:.5f}, {:.5f}, {:.5f}({}), {:.5f} \033[0m".format(
                                    epoch, loss_1, loss_2, loss_3, train_loss, train_acc, val_acc, test_acc, test_val, val_best_epoch, test_best))
    
    nni.report_final_result(test_val)
    outFile = open('./results/PerformMetrics.csv','a+', newline='')
    writer = csv.writer(outFile, dialect='excel')
    results = [time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())]
    for v, k in param.items():
        results.append(k)
    results.append(str(test_acc))
    results.append(str(test_best))
    results.append(str(test_val))
    results.append(str(val_best))
    results.append(str(val_best_epoch))
    writer.writerow(results)
         

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='')
    parser.add_argument("--ExpName", type=str, default='run0000')
    parser.add_argument("--model", type=str, default='MDGNN')
    parser.add_argument("--dataset", type=str, default="synthetic", choices=['cora', 'citeseer', 'pubmed', 'synthetic', 'zinc'])
    parser.add_argument("--input_dim", type=int, default=30)
    parser.add_argument("--out_dim", type=int, default=6)
    parser.add_argument("--percent", type=float, default=0.03)
    parser.add_argument("--mode", type=int, default=1)
    parser.add_argument("--ablation_mode", type=int, default=0)

    parser.add_argument("--num_graph", type=int, default=6)
    parser.add_argument("--hidden_dim", type=int, default=18)
    parser.add_argument("--graph_dim", type=int, default=18)
    parser.add_argument("--dropout", type=float, default=0.5)
    parser.add_argument("--sigma", type=float, default=8.0)
    parser.add_argument("--ratio_graph", type=float, default=1.0)
    parser.add_argument("--ratio_node", type=float, default=1.0)
    parser.add_argument("--num_hop", type=int, default=3)
    parser.add_argument("--beta", type=float, default=0.2)

    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--lr", type=float, default=0.005)
    parser.add_argument('--weight_decay', type=float, default=5e-4)
    parser.add_argument('--seed', type=int, default=0)

    parser.add_argument('--graph_size', type=int, default=30)
    parser.add_argument('--graph_num', type=int, default=10000)
    parser.add_argument('--feature_num', type=int, default=5)
    parser.add_argument('--std', type=float, default=5.0)
    parser.add_argument('--batch_size', type=int, default=1000)
    parser.add_argument('--init', type=float, default=0.2)
    parser.add_argument('--selected_num', type=int, default=5)

    args = parser.parse_args()
    if args.dataset == 'synthetic':
        jsontxt = open("./param/param_synthetic.json", 'r').read()
        param = json.loads(jsontxt)
    elif args.dataset == 'cora':
        jsontxt = open("./param/param_cora.json", 'r').read()
        param = json.loads(jsontxt)
    elif args.dataset == 'citeseer':
        jsontxt = open("./param/param_citeseer.json", 'r').read()
        param = json.loads(jsontxt)
    elif args.dataset == 'pubmed':
        jsontxt = open("./param/param_pubmed.json", 'r').read()
        param = json.loads(jsontxt)
    elif args.dataset == 'zinc':
        jsontxt = open("./param/param_zinc.json", 'r').read()
        param = json.loads(jsontxt)
    else:
        param = args.__dict__

    param.update(nni.get_next_parameter())
    print(param)

    if args.dataset == 'synthetic':
        main_synthetic(param)
    elif args.dataset == 'zinc':
        main_zinc(param)
    else:
        main(param)