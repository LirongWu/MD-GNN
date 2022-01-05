import os
import scipy
import heapq
import random
import warnings
import collections
import numpy as np
import seaborn as sns
import networkx as nx
import matplotlib.pyplot as plt

from PIL import Image
from sklearn.metrics import f1_score
from scipy.optimize import linear_sum_assignment

import torch
from torch.utils.data import DataLoader
import dgl

from model import get_model
from dataset import load_data

warnings.filterwarnings("ignore", category=Warning)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def set_seed(seed):
    
    np.random.seed(seed)
    random.seed(seed)
    
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    dgl.random.seed(seed)


def evaluate_f1(logits, labels):
    y_pred = torch.where(logits > 0.0, torch.ones_like(logits), torch.zeros_like(logits))
    y_pred = y_pred.detach().cpu().numpy()
    y_true = labels.detach().cpu().numpy()
    return f1_score(y_true, y_pred, average='micro')


def plot_feature_att(target, score, path):
    f = plt.figure(figsize=(4, 1))

    ax1 = plt.subplot(2, 1, 1)
    sns.heatmap(target, cmap='Blues', cbar=False, vmin=0.0, vmax=1.0)
    ax1.spines['top'].set_visible(True)
    ax1.spines['right'].set_visible(True)
    ax1.spines['bottom'].set_visible(True)
    ax1.spines['left'].set_visible(True)
    plt.xticks([])
    plt.yticks([])

    ax2 = plt.subplot(2, 1, 2)
    sns.heatmap(score, cmap='Blues', cbar=False, vmin=0.0, vmax=1.0, cbar_kws={"orientation": "horizontal"})
    ax2.spines['top'].set_visible(True)
    ax2.spines['right'].set_visible(True)
    ax2.spines['bottom'].set_visible(True)
    ax2.spines['left'].set_visible(True)
    plt.xticks([])
    plt.yticks([])

    plt.savefig(path, format='png', dpi=300, bbox_inches='tight')
    plt.close() 


def get_att(score, idx, feature_num, selected_num):

    src = np.zeros((score.shape[0], ))
    index = heapq.nlargest(selected_num, range(len(score)), score.take)
    src[index] = 1.0

    out = np.where(src[idx*feature_num:(idx+1)*feature_num] == 1.0)[0].shape[0]

    return out

    
def evaluate_att(score, param, path, plot):

    num_graph = param['num_graph']
    feature_num = param['feature_num']
    selected_num = param['selected_num']
    cost = np.zeros((num_graph, num_graph))

    for i in range(num_graph):
        for j in range(num_graph):
            cost[i, j] = get_att(score[i], j, feature_num, selected_num)
    
    row_ind, col_ind = linear_sum_assignment(feature_num-cost)
    out = cost[row_ind, col_ind].sum()

    if plot:
        sort_index = np.arange(num_graph * feature_num)
        np.random.shuffle(sort_index)

        for i in range(num_graph):
            target = np.zeros((1, score.shape[1]))
            target[:, i*feature_num:(i+1)*feature_num] = 1
            target = target[:, sort_index]

            src = np.zeros((score.shape[1], ))
            feat_index = np.where(col_ind == i)[0][0]
            index = heapq.nlargest(selected_num, range(len(score[feat_index])), score[feat_index].take)
            src[index] = 1.0

            n_min = np.min(score[feat_index, index])
            n_max = np.max(score[feat_index, index])
            att_score = (score[feat_index] - n_min) / (n_max - n_min) * 0.5 + 0.5
            att_score = (src * att_score).reshape((1, score.shape[1]))
            att_score = att_score[:, sort_index]

            os.makedirs(path, exist_ok=True)
            plot_feature_att(target, att_score, path='{}Feature_Factor_{}.png'.format(path, i))

    return out


def collate(samples):

    graphs, labels, gt_adjs = map(list, zip(*samples))
    batched_graphs = dgl.batch(graphs)

    return batched_graphs, torch.cat(tuple(labels), 0), gt_adjs


def dgl_to_adj(dgl_graph):

    adjs_list = []

    for i in range(16):
        if f'factor_{i}' not in dgl_graph.edata:
            break
        
        srt, dst = dgl_graph.edges()
        esge_weights = dgl_graph.edata[f'factor_{i}'].squeeze()
        srt, dst = srt.detach().cpu().numpy(), dst.detach().cpu().numpy()
        esge_weights = esge_weights.detach().cpu().numpy()
        
        num_node = dgl_graph.number_of_nodes()
        adjs = np.zeros((num_node, num_node))

        adjs[srt, dst] = esge_weights
        adjs += np.transpose(adjs)
        adjs /= 2.0
        adjs_list.append(adjs)
    
    return adjs_list


def translate_gt_graph_to_adj(gt_graph):
    gt_adjs = []
    gt_g_list = dgl.unbatch(gt_graph)

    for gt_g in gt_g_list:
        gt_list = []
        gt_ids = []

        n_node = gt_g.number_of_nodes()
        srt, dst = gt_g.edges()
        srt, dst = srt.detach().cpu().numpy(), dst.detach().cpu().numpy()
        edge_factor = gt_g.edata['feat'].detach().cpu().numpy()
        assert srt.shape[0] == edge_factor.shape[0]

        for edge_id in set(edge_factor):
            org_g = np.zeros((n_node, n_node))
            edge_factor_edge_id = np.zeros_like(edge_factor)
            idx = np.where(edge_factor == edge_id)[0] 
            edge_factor_edge_id[idx] = 1.0
            org_g[srt, dst] = edge_factor_edge_id
            gt_list.append(org_g)
            gt_ids.append(edge_id)

        gt_adjs.append((gt_list, gt_ids))

    return gt_adjs


class compute_GED():
    def __init__(self):
        pass
    
    def match_num_edges(self, gt_adj, pred_adj):

        np.fill_diagonal(gt_adj, 0.0)
        np.fill_diagonal(pred_adj, 0.0)

        n_edges = int(np.sum(gt_adj))
        pred_adj = pred_adj.flatten()
        
        idx = np.argpartition(-pred_adj, int(n_edges*1.0), axis=-1)
        idx = idx[:n_edges]

        pred_adj *= 0.0
        pred_adj[idx] += 1.0
        pred_adj = pred_adj.reshape((gt_adj.shape[0], gt_adj.shape[1]))

        return gt_adj, pred_adj

    def get_GED(self, gt, pred):
        gt = self.convert_to_nx(gt)
        pred = self.convert_to_nx(pred)
        
        gt_adj = nx.to_numpy_array(gt)
        pred_adj = nx.to_numpy_array(pred)
        
        np.fill_diagonal(gt_adj, 0.0)
        np.fill_diagonal(pred_adj, 0.0)

        gt_adj, pred_adj = self.match_num_edges(gt_adj, pred_adj)

        sum_adj = gt_adj + pred_adj
        sum_adj = sum_adj.reshape((-1, 1))
        indices = np.where(sum_adj == 1.0)[0] 

        return indices.shape[0]

    def convert_to_nx(self, g):
        if isinstance(g, nx.Graph):
            pass
        elif isinstance(g, dgl.DGLGraph):
            g = g.to_networkx()
        elif isinstance(g, np.ndarray):
            g = nx.DiGraph(g)
        else:
            raise NameError('unknow format of input graph')
        return g

    def hungarian_match(self, gt_list, pred_list, sample_n, path, plot=False):
        if path == 'zinc':
            gt_list, gt_idx = gt_list
        else:
            gt_idx = [g[1] for g in gt_list]
            gt_list = [g[0] for g in gt_list]

        cost = np.zeros((len(gt_list), len(pred_list)))
        for gt_i, gt in enumerate(gt_list):
            for pred_i, pred in enumerate(pred_list):
                cost[gt_i, pred_i] = self.get_GED(gt, pred)
        
        row_ind, col_ind = linear_sum_assignment(cost)

        factor_map = collections.defaultdict(list)
        for r, c in zip(row_ind, col_ind):
            factor_map[gt_idx[r]].append(c)
        
        total_ED = cost[row_ind, col_ind].sum()

        # Plot Graphs
        if plot:
            union_adj = np.zeros((30, 30))
            for i, (r, c) in enumerate(zip(row_ind, col_ind)):
                gt = gt_list[r]
                pred = pred_list[c]

                gt = self.convert_to_nx(gt)
                pred = self.convert_to_nx(pred)
                
                gt_adj = nx.to_numpy_array(gt)
                pred_adj = nx.to_numpy_array(pred)
                
                np.fill_diagonal(gt_adj, 0.0)
                np.fill_diagonal(pred_adj, 0.0)

                union_adj += gt_adj
                gt_adj, pred_adj = self.match_num_edges(gt_adj, pred_adj)

                vis_graph(gt_adj, save_name = f"{path}figs/sample{sample_n:04d}_gt-{i:01d}")
                vis_graph(pred_adj, gt_adj, pred_adj, save_name = f"{path}figs/sample{sample_n:04d}_pred-{i:01d}")
                
            vis_graph(union_adj, save_name = f"{path}figs/sample{sample_n:04d}_union_N{int(total_ED)}")

            image_concat(sample_n, total_ED, path)

        return total_ED, factor_map


def image_concat(sample_n, total_ED, path, w=1548, h=1168):
    concat_img = np.ones((3*h, 4*w, 4)) * 255
    concat_img[h:2*h, 0:w] = np.array(Image.open(f"{path}figs/sample{sample_n:04d}_union_N{int(total_ED)}.png"))
    concat_img[0:h, w:2*w] = np.array(Image.open(f"{path}figs/sample{sample_n:04d}_gt-0.png"))
    concat_img[h:2*h, w:2*w] = np.array(Image.open(f"{path}figs/sample{sample_n:04d}_gt-1.png"))
    concat_img[2*h:3*h, w:2*w] = np.array(Image.open(f"{path}figs/sample{sample_n:04d}_gt-2.png"))
    concat_img[0:h, 2*w:3*w] = np.array(Image.open(f"{path}figs/sample{sample_n:04d}_pred-0.png"))
    concat_img[h:2*h, 2*w:3*w] = np.array(Image.open(f"{path}figs/sample{sample_n:04d}_pred-1.png"))
    concat_img[2*h:3*h, 2*w:3*w] = np.array(Image.open(f"{path}figs/sample{sample_n:04d}_pred-2.png"))
    concat_img[0:h, 3*w:4*w] = np.array(Image.open(f"{path}figs/sample{sample_n:04d}_pred-0_plot.png"))
    concat_img[h:2*h, 3*w:4*w] = np.array(Image.open(f"{path}figs/sample{sample_n:04d}_pred-1_plot.png"))
    concat_img[2*h:3*h, 3*w:4*w] = np.array(Image.open(f"{path}figs/sample{sample_n:04d}_pred-2_plot.png"))
    concat_img = Image.fromarray(np.uint8(concat_img))
    concat_img.save(f"{path}vis/sample{sample_n:04d}_N{int(total_ED)}.png")


def compute_consistant(total_factor_map):

    scores = []

    for idx in total_factor_map.keys():
        inds = total_factor_map[idx]
        most_id = max(set(inds), key = inds.count)
        scores.append(float(inds.count(most_id)) / len(inds))

    return np.mean(scores)


def evaluate_graph(saved_model, seed=0):
    param = saved_model['param']
    param['seed'] = seed
    set_seed(seed)
    os.makedirs("./log/{}/Graph/SEED{}/figs/".format(param['ExpName'], seed), exist_ok=True)
    os.makedirs("./log/{}/Graph/SEED{}/vis/".format(param['ExpName'], seed), exist_ok=True)
    GED_ins = compute_GED()

    if param['dataset'] == 'zinc':
        zinc_data = load_data(param)
        train_loader = DataLoader(zinc_data.train, batch_size=1000, shuffle=True, collate_fn=zinc_data.collate)
        val_loader = DataLoader(zinc_data.val, batch_size=1000, shuffle=False, collate_fn=zinc_data.collate)
        test_loader = DataLoader(zinc_data.test, batch_size=1000, shuffle=False, collate_fn=zinc_data.collate)
        batch_graphs, batch_targets, batch_snorm_n = next(iter(test_loader))
        batch_x = batch_graphs.ndata['feat'].to(device)
        batch_targets = batch_targets.to(device)
        batch_snorm_n = batch_snorm_n.to(device)
    else:
        data = load_data(param)
        test_data = data[int(len(data)*0.8):]
        test_loader = DataLoader(test_data, batch_size=2000, shuffle=False, collate_fn=collate)
        g, labels, gt_adjs = next(iter(test_loader))

    model = get_model(param).to(device)
    model.load_state_dict(saved_model['model_state_dict'])
    model.eval()

    if param['dataset'] == 'zinc':
        model.g = batch_graphs
        _ = model.forward(batch_x, batch_snorm_n)
        pred_adjs = dgl.unbatch(model.get_factor()[0])
        gt_adjs = translate_gt_graph_to_adj(batch_graphs)
    else:
        model.g = g
        features = g.ndata['feat'].to(device)
        _ = model(features)
        pred_adjs = dgl.unbatch(model.get_factor()[0])
    
    total_GED = []
    total_factor_map = collections.defaultdict(list)
    sample_n = 0

    for gt_list, pred_g in zip(gt_adjs, pred_adjs):
        pred_list = dgl_to_adj(pred_g)
        if param['dataset'] != 'zinc':
            GED, factor_map = GED_ins.hungarian_match(gt_list, pred_list, sample_n, path="./log/{}/Graph/SEED{}/".format(param['ExpName'], seed))
        else:
            GED, factor_map = GED_ins.hungarian_match(gt_list, pred_list, sample_n, path='zinc')
        
        for idx in factor_map.keys():
            total_factor_map[idx] += factor_map[idx]
        
        total_GED.append(GED / len(gt_list))
        sample_n += 1

    c_score = compute_consistant(total_factor_map)

    if param['dataset'] != 'zinc':
        score = 1 - np.array(total_GED)
        index_list = heapq.nlargest(10, range(len(score)), score.take)

        sample_n = 0
        for gt_list, pred_g in zip(gt_adjs, pred_adjs):
            
            if sample_n in index_list:
                pred_list = dgl_to_adj(pred_g)
                GED, factor_map = GED_ins.hungarian_match(gt_list, pred_list, sample_n, path="./log/{}/Graph/SEED{}/".format(param['ExpName'], seed), plot=True)
                print("{} Finished!".format(sample_n))

            sample_n += 1

    print(f"c_score {c_score:.3f} | GED: {np.mean(total_GED):.3f} $\pm$ {np.std(total_GED):.3f}")

    return c_score, np.mean(total_GED), np.std(total_GED)


def vis_graph(g, gt_adj=None, pred_adj=None, title="", save_name=None):
    if isinstance(g, nx.Graph):
        pass
    elif isinstance(g, np.ndarray):
        g = nx.DiGraph(g)
    elif isinstance(g, dgl.DGLGraph):
        g = g.to_networkx()
    else:
        raise NameError('unknow format of input graph')

    g = nx.Graph(g)
    g = nx.DiGraph(g)
    g = nx.to_numpy_matrix(g)
    np.fill_diagonal(g, 0.0)
    g = nx.DiGraph(g)
    g.remove_nodes_from(list(nx.isolates(g)))

    if 'pred' not in save_name:
        if 'union' in save_name:   
            nx.draw_networkx(g, pos=nx.random_layout(g), arrows=False, with_labels=False, node_color="#fbb034", node_size=400, width=2.0)
        else:
            nx.draw_networkx(g, arrows=False, with_labels=False, node_color="#fbb034", node_size=400, width=2.0)
    else:
        edgelist_false1 = []
        edgelist_false2 = []
        edgelist_true = []

        for i in range(pred_adj.shape[0]):
            flag = 0

            for j in range(gt_adj.shape[1]):
                if gt_adj[i, j] == 1 and pred_adj[i, j] == 1:
                    edgelist_true.append((i, j))
                    flag = 1
                if gt_adj[i, j] == 0 and pred_adj[i, j] == 1:
                    edgelist_false1.append((i, j))
                    flag = 1
                if gt_adj[i, j] == 1 and pred_adj[i, j] == 0:
                    edgelist_false2.append((i, j))
                    flag = 1
            
            if flag == 1 and i not in g.nodes:
                g.add_node(i)

        pos = nx.spring_layout(g, iterations=30)

        plt.figure()
        nx.draw_networkx_nodes(g, pos=pos, nodelist=g.nodes, node_color="#fbb034", node_size=400)
        if len(edgelist_false1) != 0:
            nx.draw_networkx_edges(g, pos=pos, edgelist=edgelist_false1, edge_color='#FA8080', width=2.0, arrows=False, with_labels=False)
        if len(edgelist_false2) != 0:
            nx.draw_networkx_edges(g, pos=pos, edgelist=edgelist_false2, edge_color='#73ABF5', width=2.0, arrows=False, with_labels=False)
        if len(edgelist_true) != 0:
            nx.draw_networkx_edges(g, pos=pos, edgelist=edgelist_true, edge_color='k', width=2.0, arrows=False, with_labels=False)
        plt.draw()
        plt.title(title)
        plt.axis('off')
        plt.savefig(f"{save_name}_plot.png", dpi=300, bbox_inches='tight')   
        plt.close()

        plt.figure()
        nx.draw_networkx_nodes(g, pos=pos, nodelist=g.nodes, node_color="#fbb034", node_size=400)
        nx.draw_networkx_edges(g, pos=pos, edgelist=edgelist_false1 + edgelist_true, edge_color='k', width=2.0, arrows=False, with_labels=False)

    plt.draw()
    plt.title(title)
    plt.axis('off')
    plt.savefig(f"{save_name}.png", dpi=300, bbox_inches='tight')   
    plt.close()
    

def evaluate_corr(saved_model):
    param = saved_model['param']
    torch.cuda.set_device(1)
    set_seed(param['seed'])
    os.makedirs("./log/{}/Corr/".format(param['ExpName']), exist_ok=True)

    data = load_data(param)
    test_data = data[int(len(data)*0.8):]
    test_loader = DataLoader(test_data, batch_size=2000, shuffle=False, collate_fn=collate)

    model = get_model(param).to(device)
    model.load_state_dict(saved_model['model_state_dict'])

    for step, (g, labels, gt_adjs) in enumerate(test_loader):
        model.eval()
        model.g = g

        features = g.ndata['feat'].to(device)
        _ = model(features)
        hidden = model.get_hidden_feature()

        matrix = hidden[1]
        correlation = np.zeros((matrix.shape[1], matrix.shape[1]))

        for i in range(matrix.shape[1]):
            for j in range(matrix.shape[1]):
                cof = scipy.stats.pearsonr(matrix[:, i], matrix[:, j])[0]
                correlation[i][j] = cof

        lt = matrix.shape[1] // 6
        sns.heatmap(np.abs(correlation), vmin=0.0, vmax=1.0, cmap="YlGnBu")
        plt.xticks((0, lt, lt*2, lt*3, lt*4, lt*5, lt*6), ('1', str(lt), str(lt*2), str(lt*3), str(lt*4), str(lt*5), str(lt*6)), rotation=-360)
        plt.yticks((0, lt, lt*2, lt*3, lt*4, lt*5, lt*6), ('1', str(lt), str(lt*2), str(lt*3), str(lt*4), str(lt*5), str(lt*6)))
        plt.savefig("./log/{}/Corr/Correlation_FirstLayer.png".format(param['ExpName']), dpi=300, bbox_inches='tight')   
        plt.close()

if __name__ == '__main__':

    path = './log/run0000/best_model.pt'
    best_model = torch.load(path)
    evaluate_corr(best_model)
    evaluate_graph(best_model)
