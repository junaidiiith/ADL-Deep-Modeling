import os
import dgl
from torch.utils.data import Dataset
from vocab_tokenizer import VocabTokenizer
from uml_data_generation import get_encoding_size
from dgl.data import DGLDataset
from stqdm import stqdm
from uml_data_generation import get_pos_neg_graphs, promptize_node
from training_utils import get_tokenization, get_embedding


from constants import DEVICE



class GenerativeUMLDataset(Dataset):
    """
    ``GenerativeUMLDataset`` class is a pytorch dataset for the generative task of UML
    """
    def __init__(self, data, tokenizer):
        """
        Args:
            data: list of triples (entity, relations, super_type)
            tokenizer: tokenizer to tokenize the data
        """
        
        super().__init__()
        self.data = data
        
        if isinstance(tokenizer, VocabTokenizer):
            self.inputs = tokenizer.batch_encode(data, return_tensors='pt', max_length='percentile')
        else:
            max_token_length = get_encoding_size(data, tokenizer)
            self.inputs = tokenizer(data, padding=True, return_tensors='pt', max_length=max_token_length, truncation=True)
        self.labels = self.inputs['input_ids'].clone()
        self.labels[self.labels == tokenizer.pad_token_id] = -100

        # print(self.labels[0].shape)

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return {
            'input_ids': self.inputs['input_ids'][idx],
            'attention_mask': self.inputs['attention_mask'][idx],
            'labels': self.labels[idx]
        }
    


class UMLNodeDataset(Dataset):
    """
    ``UMLNodeDataset`` class is a pytorch dataset for the classification task of UML
    """
    def __init__(self, data, tokenizer, label_map):
        super().__init__()
        self.data = data
        entity_inputs = [i[0] for i in data]
        entity_labels = [i[1] for i in data]
        if isinstance(tokenizer, VocabTokenizer):
            self.inputs = tokenizer.batch_encode(entity_inputs, return_tensors='pt', max_length='percentile')
        else:
            max_token_length = get_encoding_size(entity_inputs, tokenizer)
            self.inputs = tokenizer(entity_inputs, padding=True, return_tensors='pt', max_length=max_token_length, truncation=True)
        self.labels = [label_map[i] for i in entity_labels]
        self.i2c = {v: k for k, v in label_map.items()}
        self.num_classes = len(label_map)
        
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return {
            'input_ids': self.inputs['input_ids'][idx],
            'attention_mask': self.inputs['attention_mask'][idx],
            'labels': self.labels[idx]
        }

class EncodingsDataset(Dataset):
    """
    ``EncodingsDataset`` class is a pytorch dataset to create a dataset from the tokenized data
    """
    def __init__(self, tokenized):
        self.tokenized = tokenized
    
    def __len__(self):
        return len(self.tokenized['input_ids'])
    
    def __getitem__(self, index):
        item = {key: val[index] for key, val in self.tokenized.items()}
        return item
    

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
        prepared_graphs = [self._prepare_graph(g) for g in stqdm(self.raw_graphs, desc='Preparing graphs')]
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
        pos_neg_graphs = get_pos_neg_graphs(g, self.test_size)        
        
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
    
