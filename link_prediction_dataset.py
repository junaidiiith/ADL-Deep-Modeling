import os
import dgl
from dgl.data import DGLDataset
from stqdm import stqdm
import torch
import numpy as np
from embeddings import get_embedding
from tokenization import get_tokenization
from uml_data_generation import promptize_node
from tqdm.auto import tqdm

from constants import DEVICE



class LinkPredictionDataset(DGLDataset):
    """
    ``LinkPredictionDataset`` class is a pytorch dataset for the link prediction task of UML

    Args:
        graphs: list of graphs
        tokenizer: tokenizer to tokenize the data
        model: language model to get the embeddings of the nodes
        split_type: train, test, unseen
        test_size: percentage of edges to be masked for testing the link prediction model
        raw_dir: directory to save the raw data
        save_dir: directory to save the processed data

    """
    def __init__(self, graphs, tokenizer, \
                 model, prefix='train', test_size=0.2, raw_dir='datasets/LP', save_dir='datasets/LP'):
        self.raw_graphs = graphs
        self.tokenizer = tokenizer
        self.model = model
        self.model.to(DEVICE)

        self.test_size = test_size
        self.prefix = prefix
        super().__init__(name='lp', raw_dir=raw_dir, save_dir=save_dir)
        
        
    def __getitem__(self, idx):
        return self.graphs[idx]
    
    def __len__(self):
        return len(self.graphs)
    
    def process(self):
        self.graphs = self._prepare()

    def _prepare(self):
        print("Cache does not exist. Preparing graphs...")
        # prepared_graphs = [self._prepare_graph(g) for g in stqdm(self.raw_graphs, desc='Preparing graphs')]
        prepared_graphs = [self._prepare_graph(g) for g in tqdm(self.raw_graphs, desc='Preparing graphs')]
        prepared_graphs = [g for g in prepared_graphs if g is not None]
        return prepared_graphs
    
    def _prepare_graph(self, g):
        """
        ``_prepare_graph`` function prepares the given graph for the link prediction task
        It tokenizes the nodes in the graph and gets the embeddings of the nodes using the language model
        It creates the positive and negative graphs for the given graph
        The graphs are created by masking the edges in the graph
        The edges are masked by setting the 'masked' attribute of the edge to True
        pos_neg_graphs is a dictionary of: 
        {
            'train_pos_g': train positive graph,
            'train_neg_g': train negative graph,
            'test_pos_g': test positive graph,
            'test_neg_g': test negative graph,
            'train_g': train graph
        }
        """
        node_strs = [promptize_node(g, n) for n in g.nodes()]
        node_encodings = get_tokenization(self.tokenizer, node_strs)

        node_embeddings = get_embedding(self.model, node_encodings)
        try:
            pos_neg_graphs = get_pos_neg_graphs(g, self.test_size)
        except Exception:
            return None
        
        dgl_graph = pos_neg_graphs['train_g']
        dgl_graph.ndata['h'] = node_embeddings

        return pos_neg_graphs

    
    def save(self):
        """Save list of DGLGraphs using DGL save_graphs."""
        print("Saving graphs to cache...")
        keys = ['train_pos_g', 'train_neg_g', 'test_pos_g', 'test_neg_g', 'train_g']
        
        graphs = {k: [g[k] for g in self.graphs] for k in keys}
        for k, v in graphs.items():
            dgl.save_graphs(os.path.join(self.save_dir, f'{self.name}_{k}_{self.prefix}.dgl'), v)
    
    
    def load(self):
        """Load list of DGLGraphs using DGL load_graphs."""
        print("Loading graphs from cache...")
        
        keys = ['train_pos_g', 'train_neg_g', 'test_pos_g', 'test_neg_g', 'train_g']
        k_graphs = {k: [] for k in keys}
        for k in keys:
            k_graphs[k] = dgl.load_graphs(os.path.join(self.save_dir, f'{self.name}_{k}_{self.prefix}.dgl'))[0]
        
        self.graphs = list()
        for i in range(len(k_graphs['train_g'])):
            self.graphs.append({k: v[i] for k, v in k_graphs.items()})
        
        print(f'Loaded {len(self.graphs)} graphs.')

        
    def has_cache(self):
        print("Checking if cache exists at: ", os.path.join(self.save_dir, f'{self.name}_train_g_{self.prefix}.dgl'))
        return os.path.exists(os.path.join(self.save_dir, f'{self.name}_train_g_{self.prefix}.dgl'))
    

def get_pos_neg_graphs(nxg, tr=0.2):

    """
    ``get_pos_neg_graphs`` function returns the positive and negative graphs for the given graph
    There are train positive, train negative, test positive and test negative graphs
    train positive graph is the graph with masked edges removed and only positive edges
    train negative graph is the graph with masked edges removed and only negative edges
    test positive graph is the graph with masked edges and only positive edges
    test negative graph is the graph with masked edges and only negative edges

    tr: test size

    positive edges are the actual edges in the graph
    negative edges are the edges that are not in the graph

    The edges are masked for testing the link prediction model
    The edges are masked by setting the 'masked' attribute of the edge to True
    The edges are unmasked by setting the 'masked' attribute of the edge to False
    """


    g = dgl.from_networkx(nxg, edge_attrs=['masked'])
    u, v = g.edges()
    test_mask = torch.where(g.edata['masked'])[0]
    train_mask = torch.where(~g.edata['masked'])[0]
    test_size = int(g.number_of_edges() * tr)
    test_pos_u, test_pos_v = u[test_mask], v[test_mask]
    train_pos_u, train_pos_v = u[train_mask], v[train_mask]

    # Find all negative edges and split them for training and testing
    adj = g.adjacency_matrix()
    adj_neg = 1 - adj.to_dense() - np.eye(g.number_of_nodes())
    neg_u, neg_v = np.where(adj_neg != 0)

    try:
        neg_eids = np.random.choice(len(neg_u), g.number_of_edges())
    except ValueError:
        neg_eids = np.random.choice(len(neg_u), g.number_of_edges(), replace=True)
        print(sum(adj.to_dense().flatten()))
        print(sum(adj_neg.flatten()))
        print(g.number_of_edges())
        print(g.number_of_nodes())

    test_neg_u, test_neg_v = neg_u[neg_eids[:test_size]], neg_v[neg_eids[:test_size]]
    train_neg_u, train_neg_v = neg_u[neg_eids[test_size:]], neg_v[neg_eids[test_size:]]

    train_g = dgl.remove_edges(g, test_mask)

    train_pos_g = dgl.graph((train_pos_u, train_pos_v), num_nodes=g.number_of_nodes())
    train_neg_g = dgl.graph((train_neg_u, train_neg_v), num_nodes=g.number_of_nodes())



    test_pos_g = dgl.graph((test_pos_u, test_pos_v), num_nodes=g.number_of_nodes())
    test_neg_g = dgl.graph((test_neg_u, test_neg_v), num_nodes=g.number_of_nodes())
    
    graphs = {
        'train_pos_g': train_pos_g,
        'train_neg_g': train_neg_g,
        'test_pos_g': test_pos_g,
        'test_neg_g': test_neg_g,
        'train_g': train_g
    }
    return graphs
