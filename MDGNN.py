import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter

import dgl
import dgl.function as fn

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class MDGNN(nn.Module):
    def __init__(self, param):
        super(MDGNN, self).__init__()

        self.g = None
        self.param = param
        self.input_dim = param['input_dim']
        self.hidden_dim = param['hidden_dim']
        self.out_dim = param['out_dim']
        self.graph_dim = param['graph_dim']
        self.num_graph = param['num_graph']
        self.dropout = param['dropout']

        self.GraphLearning = GraphLearning(self.input_dim, self.graph_dim, self.num_graph, param)

        self.layers = nn.ModuleList()
        self.layers.append(GCN(self.input_dim, self.hidden_dim, self.num_graph, nn.LeakyReLU(negative_slope=0.2), param))
        if param['dataset'] == 'synthetic':
            self.layers.append(GCN(self.hidden_dim * self.num_graph, self.out_dim, self.num_graph, nn.LeakyReLU(negative_slope=0.2), param))
        elif param['dataset'] == 'zinc':
            self.activate = torch.nn.LeakyReLU(negative_slope=0.2)
            self.embedding = nn.Embedding(28, self.input_dim)

            self.layers.append(GCN(self.input_dim, self.hidden_dim, self.num_graph, None, param))
            self.layers.append(GCN(self.hidden_dim * self.num_graph, self.hidden_dim, self.num_graph, None, param))
            self.layers.append(GCN(self.hidden_dim * self.num_graph, self.hidden_dim, self.num_graph, None, param))
            self.layers.append(GCN(self.hidden_dim * self.num_graph, self.hidden_dim, self.num_graph, None, param))

            self.regressor1 = nn.Linear(self.hidden_dim * self.num_graph, self.hidden_dim).to(device)
            self.regressor2 = nn.Linear(self.hidden_dim, 1).to(device)

            self.BNs = nn.ModuleList()
            self.BNs.append(nn.BatchNorm1d(self.hidden_dim * self.num_graph))
            self.BNs.append(nn.BatchNorm1d(self.hidden_dim * self.num_graph))
            self.BNs.append(nn.BatchNorm1d(self.hidden_dim * self.num_graph))
            self.BNs.append(nn.BatchNorm1d(self.hidden_dim * self.num_graph))
        else:
            self.layers.append(GCN(self.hidden_dim * self.num_graph, self.out_dim, 1, nn.LeakyReLU(negative_slope=0.2), param))
        self.linear = nn.Linear(self.out_dim * self.num_graph, self.out_dim).to(device)

    def forward(self, features, snorm_n=None, mode='train'):
        if self.param['dataset'] == 'zinc':
            features = self.embedding(features)
            self.g = self.GraphLearning(self.g, features)
            self.feature_list = [features]

            for layer, bn in zip(self.layers[1:], self.BNs):
                if mode == 'train':
                    features = torch.nn.functional.dropout(features, self.dropout)
                else:
                    features = torch.nn.functional.dropout(features, 0.0)
                features = layer(self.g, features)
                features = features * snorm_n
                features = bn(features)
                features = self.activate(features)
                self.feature_list.append(features.detach().cpu().numpy())

            if mode == 'train':
                features = torch.nn.functional.dropout(features, self.dropout)
            else:
                features = torch.nn.functional.dropout(features, 0.0)

            self.g.ndata['h'] = features
            features = dgl.mean_nodes(self.g, 'h')
            features = torch.relu(features)
            features = self.regressor1(features)
            features = torch.relu(features)
            features = self.regressor2(features)

            return features

        self.g = self.GraphLearning(self.g, features)
        self.feature_list = [features]

        for layer in self.layers:
            if mode == 'train':
                features = torch.nn.functional.dropout(features, self.dropout)
            else:
                features = torch.nn.functional.dropout(features, 0.0)
            features = layer(self.g, features)
            self.feature_list.append(features.detach().cpu().numpy())

        if self.param['dataset'] == 'synthetic':
            if mode == 'train':
                features = torch.nn.functional.dropout(features, self.dropout)
            else:
                features = torch.nn.functional.dropout(features, 0.0)

            self.g.ndata['h'] = features
            features = dgl.mean_nodes(self.g, 'h')
            features = torch.tanh(features)
            features = self.linear(features)

            return features
        else:
            return features

    def compute_disentangle_loss(self):
        loss_graph, node_loss = self.GraphLearning.compute_disentangle_loss(self.g)
        return loss_graph, node_loss

    def get_factor(self):
        factor_list = [self.g]
        return factor_list

    def get_hidden_feature(self):
        return self.feature_list


class NodeApplyModule(nn.Module):
    def __init__(self, input_dim, output_dim, activation):
        super(NodeApplyModule, self).__init__()
        self.linear = nn.Linear(input_dim, output_dim).to(device)
        self.activation = activation

    def forward(self, node_features):
        h = self.linear(node_features)
        if self.activation is not None:
            h = self.activation(h)
        return h


class GCN(nn.Module):
    def __init__(self, input_dim, output_dim, num_graph, activation, param):
        super(GCN, self).__init__()
        self.param = param
        self.num_graph = num_graph
        self.apply_mod = nn.ModuleList()
        for num in range(self.num_graph):
            self.apply_mod.append(NodeApplyModule(input_dim, output_dim, activation))

    def forward(self, g, features):
        out_features = []
        norm = torch.pow(g.in_degrees().float().clamp(min=1), -0.5).view(-1, 1).to(features.device)

        for num in range(self.num_graph):
            g.ndata.update({f'feature_{num}_0': features})

            for k in range(self.param['num_hop']):
                hidden = g.ndata[f'feature_{num}_{k}']
                g.ndata[f'feature_{num}_{k+1}'] = hidden * norm
                g.update_all(fn.u_mul_e(f'feature_{num}_{k+1}', f"factor_{num}", 'm'), fn.sum('m', f'feature_{num}_{k+1}'))
                g.ndata[f'feature_{num}_{k+1}'] = g.ndata[f'feature_{num}_{k+1}'] * (1.0 - self.param['beta']) + g.ndata[f'feature_{num}_{0}'] * self.param['beta']

            last_one = self.param['num_hop']
            out = self.apply_mod[num](g.ndata[f'feature_{num}_{last_one}'])
            out_features.append(out)

        out = torch.cat(tuple([rst for rst in out_features]), -1)

        return out
            

class GraphLearning(nn.Module):
    def __init__(self, input_dim, graph_dim, num_graph, param):
        super(GraphLearning, self).__init__()
        self.num_graph = num_graph
        self.param = param

        self.linear = nn.ModuleList()
        for num in range(self.num_graph):
            self.linear.append(nn.Linear(input_dim, graph_dim//num_graph).to(device))

        self.att_ls = nn.ModuleList()
        self.att_rs = nn.ModuleList()
        for num in range(self.num_graph):
            self.att_ls.append(nn.Linear(graph_dim//num_graph, 1).to(device))
            self.att_rs.append(nn.Linear(graph_dim//num_graph, 1).to(device))

        self.att = Parameter(torch.Tensor(num_graph, input_dim))
        if param['dataset'] == 'synthetic':
            torch.nn.init.uniform(self.att, a=param['init'], b=param['init'])
        else:
            torch.nn.init.xavier_normal_(self.att)
        
        graph_dim_div = graph_dim // num_graph * num_graph
        self.GraphAE = GraphEncoder(graph_dim_div, graph_dim_div // 2).to(device)
        if self.param['mode'] == 0:
            self.classifier = nn.Linear(graph_dim_div, num_graph+1).to(device)
            self.loss_fn = nn.CrossEntropyLoss()
        elif self.param['mode'] == 1:
            self.classifier = nn.Linear(graph_dim_div, num_graph+1).to(device)
            self.loss_fn = nn.MSELoss()
        elif self.param['mode'] == 2:
            self.classifier = nn.Linear(graph_dim_div, 1).to(device)
            self.loss_fn = nn.MSELoss()

    def forward(self, g, features):
        
        hidden_list = []
        for num in range(self.num_graph):
            features_att = features * self.att[num:num+1, :]
            hidden = self.linear[num](features_att)
            hidden_list.append(hidden)
            
            a_l = self.att_ls[num](hidden)
            a_r = self.att_rs[num](hidden)
            g.ndata.update({f'a_l_{num}': a_l, f'a_r_{num}': a_r})
            g.apply_edges(fn.u_add_v(f'a_l_{num}', f'a_r_{num}', f"factor_{num}"))
            g.edata[f"factor_{num}"] = torch.sigmoid(self.param['sigma'] * g.edata[f"factor_{num}"])
        
        self.hidden = torch.cat(tuple(hidden_list), -1)
        return g

    def compute_disentangle_loss(self, g):
        factors_feature = [self.GraphAE(g, self.hidden, f"factor_{num}") for num in range(self.num_graph)] 
        factors_feature.append(self.GraphAE(g, self.hidden, "normal"))
        labels = [torch.ones(f.shape[0])*i for i, f in enumerate(factors_feature)]
        labels = torch.cat(tuple(labels), 0).long().to(device)

        factors_feature = torch.cat(tuple(factors_feature), 0)
        pred = self.classifier(factors_feature)
        if self.param['mode'] == 0:
            pred = nn.Softmax(dim=1)(pred)
            loss_graph = self.loss_fn(pred, labels)
        else:
            loss_graph_list = []
            for i in range(self.num_graph+1):
                for j in range(i+1, self.num_graph+1):
                    loss_graph_list.append(self.loss_fn(pred[i], pred[j]))
            loss_graph_list = torch.Tensor(loss_graph_list)
            loss_graph = -torch.sum(loss_graph_list)

        node_loss = torch.norm(torch.mm(self.att, self.att.t()) * (1-torch.eye(self.num_graph).to(self.att.device))) ** 2

        return loss_graph, node_loss


class GraphEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(GraphEncoder, self).__init__()
        self.apply_mod1 = NodeApplyModule(input_dim, hidden_dim, F.tanh)
        self.apply_mod2 = NodeApplyModule(hidden_dim, input_dim, F.tanh)

    def forward(self, g, features, factor_key):
        g = g.local_var()

        norm = torch.pow(g.in_degrees().float().clamp(min=1), -0.5).view(-1, 1).to(features.device)

        g.ndata.update({'h': features * norm})
        if "factor" in factor_key:
            g.update_all(fn.u_mul_e('h', factor_key, 'm'), fn.sum('m', 'h'))
        else:
            g.update_all(fn.copy_src(src="h",out="m"), fn.sum('m', 'h'))
        features = self.apply_mod1(g.ndata['h'])

        g.ndata.update({'h': features * norm})
        if "factor" in factor_key:
            g.update_all(fn.u_mul_e('h', factor_key, 'm'), fn.sum('m', 'h'))
        else:
            g.update_all(fn.copy_src(src="h",out="m"), fn.sum('m', 'h'))
        g.ndata['h'] = self.apply_mod2(g.ndata['h'])

        h = dgl.mean_nodes(g, 'h')

        return h
