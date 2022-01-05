import os
import dgl
import csv
import time
import torch
import pickle
import random
import numpy as np
import networkx as nx

from dgl import DGLGraph
from dgl.data import citation_graph

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def load_data(param):
    if param['percent'] == 0:
        public = 1
    else:
        public = 0
    if param['dataset'] == 'cora':
        data = citation_graph.load_cora(public, param['percent'])
    if param['dataset'] == 'citeseer':
        data = citation_graph.load_citeseer(public, param['percent'])
    if param['dataset'] == 'pubmed':
        data = citation_graph.load_pubmed(public, param['percent'])
    if param['dataset'] == 'synthetic':
        synthetic_graph = load_synthetic(param)
        data = synthetic_graph.generate()  
        return data
    if param['dataset'] == 'zinc':
        zinc_data = MoleculeDatasetDGL()
        return zinc_data

    features = torch.FloatTensor(data.features)
    labels = torch.LongTensor(data.labels)
    train_mask = torch.BoolTensor(data.train_mask)
    val_mask = torch.BoolTensor(data.val_mask)
    test_mask = torch.BoolTensor(data.test_mask)

    g = data.graph
    g.remove_edges_from(nx.selfloop_edges(g))
    g = DGLGraph(g)
    g.add_edges(g.nodes(), g.nodes())

    return g, features.to(device), labels.to(device), train_mask.to(device), val_mask.to(device), test_mask.to(device)


class load_synthetic:
    def __init__(self, param):
        self.num_graph = param['num_graph']
        self.graph_size = param['graph_size']
        self.graph_num = param['graph_num']
        self.feature_num = param['feature_num']
        self.std = param['std']
        self.seed = param['seed']
        self.saved_file = f'./data/synthetic/synthetic_graph_{self.num_graph}_{self.seed}.pkl'
        os.makedirs(os.path.dirname(self.saved_file), exist_ok=True)
        
    def generate(self):
        graph_size = self.graph_size
        num_graph = self.graph_num

        if os.path.isfile(self.saved_file):
            print(f"load synthetic graph from {self.saved_file}")
            with open(self.saved_file, 'rb') as f:
                return pickle.load(f)
        
        graph_list = load_synthetic.get_graph_list(self.num_graph)
        samples = []

        for _ in range(num_graph):
            union_graph = np.zeros((graph_size, graph_size))
            labels = np.zeros((1, len(graph_list)))
            features = np.random.normal(size=(graph_size, self.num_graph*self.feature_num), scale=self.std, loc=0)

            factor_graphs = []
            idx_list = list(range(len(graph_list)))
            random.shuffle(idx_list)
            
            for i in range((len(idx_list) + 1) // 2):
                idx = idx_list[i]
                labels[0, idx] = 1

                single_graph = graph_list[idx]
                padded_graph = np.zeros((graph_size, graph_size))
                padded_graph[:single_graph.shape[0], :single_graph.shape[0]] = single_graph
                
                random_index = np.arange(padded_graph.shape[0])
                np.random.shuffle(random_index)
                padded_graph = padded_graph[random_index]
                padded_graph = padded_graph[:, random_index]

                padded_feature = np.random.normal(size=(graph_size, self.feature_num), scale=self.std, loc=0)
                padded_feature[:single_graph.shape[0], :] = np.random.normal(size=(single_graph.shape[0], self.feature_num), scale=self.std, loc=idx+1)
                features[:, idx*self.feature_num:(idx+1)*self.feature_num] = padded_feature[random_index]

                union_graph += padded_graph
                factor_graphs.append((padded_graph, idx))

            g = dgl.DGLGraph()
            g.from_networkx(nx.DiGraph(union_graph))
            g = dgl.transform.add_self_loop(g)
            g.ndata['feat'] = torch.tensor(features).float()
            labels = torch.tensor(labels)
            samples.append((g, labels, factor_graphs))

        with open(self.saved_file, 'wb') as f:
            pickle.dump(samples, f)
            print(f"dataset saved to {self.saved_file}")
            
        return samples


    @staticmethod
    def get_graph_list(num_graph):
        graph_list = []

        g = nx.turan_graph(n=5, r=2)
        graph_list.append(nx.to_numpy_array(g))

        g = nx.house_x_graph()
        graph_list.append(nx.to_numpy_array(g))
        
        g = nx.balanced_tree(r=3, h=2)
        graph_list.append(nx.to_numpy_array(g))
        
        g = nx.grid_2d_graph(m=3, n=3)
        graph_list.append(nx.to_numpy_array(g))

        g = nx.hypercube_graph(n=3)
        graph_list.append(nx.to_numpy_array(g))

        g = nx.octahedral_graph()
        graph_list.append(nx.to_numpy_array(g))

        return graph_list[:num_graph]


class MoleculeDGL(torch.utils.data.Dataset):
    def __init__(self, data_dir, split, num_graphs):
        self.data_dir = data_dir
        self.split = split
        self.num_graphs = num_graphs
        self.file_path = data_dir + f'/graph_list_labels_{self.split}.pt'
        
        
        if not os.path.isfile(self.file_path):
            with open(data_dir + "/%s.pickle" % self.split, "rb") as f:
                self.data = pickle.load(f)

            with open(data_dir + "/%s.index" % self.split, "r") as f:
                data_idx = [list(map(int, idx)) for idx in csv.reader(f)]
                self.data = [self.data[i] for i in data_idx[0]]
                
            assert len(self.data) == num_graphs, "Sample num_graphs again; available idx: train/val/test => 10k/1k/1k"
        
        self.graph_lists = []
        self.graph_labels = []
        self._prepare()
        self.n_samples = len(self.graph_lists)
        
    def _prepare(self):
        if os.path.isfile(self.file_path):
            print(f"load from {self.file_path}")
            with open(self.file_path, 'rb') as f:
                self.graph_lists, self.graph_labels = pickle.load(f)
            return

        print("preparing %d graphs for the %s set..." % (self.num_graphs, self.split.upper()))
        
        for molecule in self.data:
            node_features = molecule['atom_type'].long()
            
            adj = molecule['bond_type']
            edge_list = (adj != 0).nonzero()
            edge_idxs_in_adj = edge_list.split(1, dim=1)
            edge_features = adj[edge_idxs_in_adj].reshape(-1).long()
            
            g = dgl.DGLGraph()
            g.add_nodes(molecule['num_atom'])
            g.ndata['feat'] = node_features
            
            for src, dst in edge_list:
                g.add_edges(src.item(), dst.item())
            g.edata['feat'] = edge_features
            
            self.graph_lists.append(g)
            self.graph_labels.append(molecule['logP_SA_cycle_normalized'])
        
        with open(self.file_path, 'wb') as f:
            pickle.dump((self.graph_lists, self.graph_labels), f)
        
    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        return self.graph_lists[idx], self.graph_labels[idx]


class MoleculeDatasetDGL(torch.utils.data.Dataset):
    def __init__(self, name='zinc'):

        data_dir = "./data/zinc/molecules"
        
        self.train = MoleculeDGL(data_dir, 'train', num_graphs=10000)
        self.val = MoleculeDGL(data_dir, 'val', num_graphs=1000)
        self.test = MoleculeDGL(data_dir, 'test', num_graphs=1000)
        
    def collate(self, samples):
        graphs, labels = map(list, zip(*samples))
        batched_graph = dgl.batch(graphs)
        labels = torch.tensor(np.array(labels)).unsqueeze(1)

        tab_sizes_n = [graphs[i].number_of_nodes() for i in range(len(graphs))]
        tab_snorm_n = [torch.FloatTensor(size,1).fill_(1./float(size)) for size in tab_sizes_n]
        snorm_n = torch.cat(tab_snorm_n).sqrt()  
        tab_sizes_e = [graphs[i].number_of_edges() for i in range(len(graphs))]
        tab_snorm_e = [torch.FloatTensor(size,1).fill_(1./float(size)) for size in tab_sizes_e]
        snorm_e = torch.cat(tab_snorm_e).sqrt()
        
        return batched_graph, labels, snorm_n