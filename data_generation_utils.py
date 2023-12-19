import os
from dgl.data import DGLDataset
import dgl
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer
from tqdm.auto import tqdm
from nltk.tokenize import word_tokenize
import torch
import numpy as np
from utils import clean_text
from models import get_embedding


SSP = "<superType>"
ESP = "</superType>"
SEN = "<entity>"
EEN = "</entity>"

SRP = "<relations>"
ERP = "</relations>"

PAD = "<pad>"
UNK = "<unk>"
SOS = "<s>"
EOS = "</s>"
MASK = "<mask>"
SEP = "<sep>"

SPECIAL_TOKENS = [PAD, UNK, SOS, EOS, MASK, SEP, SSP, ESP, SEN, EEN, SRP, ERP]



promptize_triple = lambda x: f"{SOS} {SSP} {clean_text(x[2])} {ESP} {SEN} {clean_text(x[0])} {EEN} {SRP} {clean_text(x[1])} {ERP} {EOS}"
promptize_node = lambda g, n: promptize_triple((n, g.nodes[n]['references'] if 'references' in g.nodes[n] else '', g.nodes[n]['super_types'] if 'super_types' in g.nodes[n] else ''))

def promptize_super_type_generation(x):
    return f"{SOS} {SEN} {clean_text(x[0])} {EEN} {SRP} {clean_text(x[1])} {ERP} {SEP} {SSP} {clean_text(x[2])} {ESP} {EOS}"

def promptize_entity_type_generation(x):
    return f"{SOS} {SSP} {clean_text(x[1])} {ESP} {SRP} {clean_text(x[1])} {ERP} {SEP} {SEN} {clean_text(x[0])} {EEN} {EOS}"

def promptize_super_type_classification(x):
    return f"{SOS} {SEN} {clean_text(x[0])} {EEN} {SRP} {clean_text(x[1])} {ERP} {EOS}", f"{clean_text(x[2])}"

def promptize_entity_type_classification(x):
    return f"{SOS} {SSP} {clean_text(x[1])} {ESP} {SRP} {clean_text(x[1])} {ERP} {EOS}", f"{clean_text(x[0])}"


def remove_duplicates(data):
    return list({str(i): i for i in data}.values())

def print_sample_data(data):
    for split_type in data:
        print(f"Split type: {split_type}")
        print(f"Total number of samples: {len(data[split_type])}")
        print(f"2 Samples: {data[split_type][:2]}")
        print()


def get_promptized_data_for_super_type_generation(data):
    promptized_data = {
        split_type: remove_duplicates([promptize_super_type_generation(i) for i in data[split_type] if len(i[2].strip())])\
              for split_type in data
    }
    # print_sample_data(promptized_data)
    
    return promptized_data

def get_promptized_data_for_entity_generation(data):
    promptized_data = {
        split_type: remove_duplicates(
            [promptize_entity_type_generation(i) for i in data[split_type] if len(i[1].strip())])\
              for split_type in data
    }
    # print_sample_data(promptized_data)
    return promptized_data

def get_promptized_data_for_super_type_classification(data):
    promptized_data = {
        split_type: remove_duplicates(
            [promptize_super_type_classification(i) for i in data[split_type] if len(i[2].strip())])\
              for split_type in data
    }
    print_sample_data(promptized_data)
    
    return promptized_data

def get_promptized_data_for_entity_classification(data):
    promptized_data = {
        split_type: remove_duplicates([promptize_entity_type_classification(i) for i in data[split_type] if len(i[1].strip())])\
              for split_type in data
    }
    print_sample_data(promptized_data)
    return promptized_data

def get_promptized_data_for_generation(data):
    data_for_super_type_generation = get_promptized_data_for_super_type_generation(data)
    data_for_entity_generation = get_promptized_data_for_entity_generation(data)

    promptized_data = {
        split_type: data_for_super_type_generation[split_type] + data_for_entity_generation[split_type]\
              for split_type in data
    }
    print_sample_data(promptized_data)
    
    return promptized_data


def get_data_for_classification(data, class_type='super'):
    if class_type == 'super':
        promptized_data = get_promptized_data_for_super_type_classification(data)
    else:
        promptized_data = get_promptized_data_for_entity_classification(data)
    return promptized_data


def get_classification_dataset(data, tokenizer, encoder, class_type='super', multi_label=False):
    if class_type == 'super':
        return get_super_type_classification_dataset(data, tokenizer, encoder, multi_label=multi_label)
    else:
        return get_entity_classification_dataset(data, tokenizer, encoder)


def get_super_type_labels(super_types, super_type_map, multi_label=False):
    stp_labels = [[super_type_map[j] for j in super_type] for super_type in super_types]
    if not multi_label:
        stp_labels = np.array([i[0] for i in stp_labels])
        stp_labels = torch.from_numpy(stp_labels)
    else:
        l = list()
        for stp_label in stp_labels:
            row = torch.zeros(len(super_type_map))
            for label in stp_label:
                row[label] = 1
            l.append(row)
            
        stp_labels = torch.stack(l)
        
    return stp_labels

def get_encoding_size(data, tokenizer):
    tokens = tokenizer(data)
    lengths = [len(i) for i in tokens['input_ids']]
    size = int(np.percentile(lengths, 99.5))
    # print("Encoding size: ", size)
    return size


def get_pretrained_lm_tokenizer(model_name, special_tokens=SPECIAL_TOKENS):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.add_special_tokens({'additional_special_tokens': special_tokens})
    if tokenizer.pad_token is None:
        tokenizer.pad_token = "<pad>"

    print("Vocab size: ", len(tokenizer))
    return tokenizer

def get_word_tokenizer_tokenizer(data, lower=True, special_tokens=SPECIAL_TOKENS):
    tokenizer = VocabTokenizer(data, lower=lower, special_tokens=special_tokens)
    print("Vocab size: ", len(tokenizer))
    return tokenizer


def get_generative_uml_dataset(data, tokenizer):
    dataset = {
        split_type: GenerativeUMLDataset(data[split_type], tokenizer) for split_type in data
    }
    return dataset

def get_super_type_classification_dataset(data, tokenizer, super_type_map, multi_label=False):
    dataset = {
        split_type: SuperTypeClassificationDataset(
            data[split_type], tokenizer, super_type_map, multi_label=multi_label) for split_type in data
    }
    return dataset

def get_entity_classification_dataset(data, tokenizer, label_map):
    dataset = {
        split_type: EntityClassificationDataset(
            data[split_type], tokenizer, label_map) for split_type in data
    }
    return dataset


def get_dataloaders(dataset, batch_size=32):
    dataloaders = {
        split_type: DataLoader(
            dataset[split_type], batch_size=batch_size, shuffle=split_type == 'train') for split_type in dataset
    }
    return dataloaders

    
def get_gpt2_tokenized_data(data, tokenizer):
    tokenized_data = {
        split_type: tokenizer(
            data[split_type], 
            padding=True, 
            return_tensors='pt', 
            max_length=get_encoding_size(data[split_type], tokenizer), 
            truncation=True
        ) for split_type in data
    }
    return tokenized_data


def get_gpt2_dataset(data, tokenizer):
    tokenized_data = get_gpt2_tokenized_data(data, tokenizer)
    dataset = {
        split_type: GPT2Dataset(tokenized_data[split_type]) for split_type in data
    }
    return dataset


def get_pos_neg_graphs(nxg, tr=0.2):
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

    neg_eids = np.random.choice(len(neg_u), g.number_of_edges())
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




def get_kfold_lm_data(data, seed=42, test_size=0.1):

    seen_graph_triples = data['train_triples']
    unseen_graph_triples = data['test_triples']

    X_train = [1]*len(seen_graph_triples)

    k_folds = int(1/test_size)
    i = 0
    skf = StratifiedKFold(n_splits=k_folds, shuffle=True, random_state=seed)

    for train_idx, test_idx in skf.split(X_train, X_train):
        seen_train = [seen_graph_triples[i] for i in train_idx]
        seen_test = [seen_graph_triples[i] for i in test_idx]
        
        print("Train graph triples: ", len(seen_train), "Test graph triples: ", len(seen_test), "Unseen graph triples: ", len(unseen_graph_triples))

        data = {
            'train': seen_train,
            'test': seen_test,
            'unseen': unseen_graph_triples,
        }
        
        i += 1
        yield data


def get_kfold_lp_data(data, seed=42, test_size=0.1):

    seen_graphs = data['train_graphs']
    unseen_graphs = data['test_graphs']

    X_train = [1]*len(seen_graphs)

    k_folds = int(1/test_size)
    i = 0
    skf = StratifiedKFold(n_splits=k_folds, shuffle=True, random_state=seed)

    for train_idx, test_idx in skf.split(X_train, X_train):
        seen_train = [seen_graphs[i] for i in train_idx]
        seen_test = [seen_graphs[i] for i in test_idx]
        

        # train, test = train_test_split(node_triples, test_size=test_size, random_state=seed)
        print("Train graphs: ", len(seen_train), "Test graphs: ", len(seen_test), "Unseen graphs: ", len(unseen_graphs))

        data = {
            'train': seen_train,
            'test': seen_test,
            'unseen': unseen_graphs,
        }
        
        i += 1
        yield data


class VocabTokenizer:
    def __init__(self, data, lower=True, special_tokens=[]):
        self.lower = lower
        self.vocab = {}
        self.special_tokens = special_tokens
        
        for i in self.special_tokens:
            self.vocab[i] = len(self.vocab)

        for split_type in data:
            for i in tqdm(data[split_type]):
                word = " ".join(i) if isinstance(i, tuple) else i
                for j in word_tokenize(clean_text(word) if not self.lower else clean_text(word).lower()):
                    if j not in self.vocab:
                        self.vocab[j] = len(self.vocab)
        
        self.pad_token_id = self.vocab[PAD]
        self.pad_token = PAD

        self.unknown_token_id = self.vocab[UNK]
        self.unknown_token = UNK
        
        self.index_to_key = {v: k for k, v in self.vocab.items()}
    
    def batch_encode(self, x, return_tensors=None, max_length=None):
        assert isinstance(x, list), "Input must be a list"
        batch_encodings = [self.encode(i) for i in tqdm(x, desc='Encoding')]
        lengths = [len(i) for i in batch_encodings]
        perc_max_length = int(np.percentile(lengths, 99.95))
        max_length = 512 if max_length is None else (perc_max_length if max_length == 'percentile' else max_length)
        max_length = min(max_length, max([len(i) for i in batch_encodings]))
        
        batch_input_ids = [i[:min(max_length, len(i))] + [self.pad_token_id] * (max_length - min(max_length, len(i))) for i in batch_encodings]
        batch_attention_mask = [[1] * min(max_length, len(i)) + [0] * (max_length - min(max_length, len(i))) for i in batch_encodings]

        if return_tensors == 'pt':
            return {
                'input_ids': torch.LongTensor(batch_input_ids),
                'attention_mask': torch.LongTensor(batch_attention_mask)
            }
        elif return_tensors == 'np':
            return {
                'input_ids': np.array(batch_input_ids),
                'attention_mask': np.array(batch_attention_mask)
            }
        else:
            return {
                'input_ids': batch_input_ids,
                'attention_mask': batch_attention_mask
            }


    def encode(self, x, return_tensors=None):
        input_ids = self(x)
        if return_tensors == 'pt':
            return torch.LongTensor(input_ids)
        elif return_tensors == 'np':
            return np.array(input_ids)
        return input_ids
    

    def __call__(self, x):
        if isinstance(x, tuple) or isinstance(x, list):
            x = " ".join(x)
        
        words, x = x.split(), list()
        for i in range(0, len(words)):
            if words[i] in self.special_tokens:
                x.append(words[i])
            else:
                x.extend(word_tokenize(clean_text(words[i]) if not self.lower else clean_text(words[i]).lower()))

        return [self.vocab.get(i, self.vocab['<unk>']) for i in x]
    
    def decode(self, x):
        assert isinstance(x, list), "Input must be a list"
        return [self.index_to_key[i] for i in x]
    
    def __len__(self):
        return len(self.vocab)
    
    def get_vocab(self):
        return self.vocab

    def add_special_tokens(self, special_tokens):
        for i in special_tokens:
            if i not in self.vocab:
                self.vocab[i] = len(self.vocab)
        self.index_to_key = {v: k for k, v in self.vocab.items()}
    
    def __str__(self) -> str:
        return f"VocabTokenizer(vocab_size={len(self.vocab)})"


class GenerativeUMLDataset(Dataset):
    def __init__(self, data, tokenizer):
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
    

class SuperTypeClassificationDataset(Dataset):
    def __init__(self, data, tokenizer, super_type_map, multi_label=False):
        super().__init__()
        self.data = data
        super_type_inputs = [i[0] for i in data]
        super_type_labels = [i[1] for i in data]
        if isinstance(tokenizer, VocabTokenizer):
            self.inputs = tokenizer.batch_encode(super_type_inputs, return_tensors='pt', max_length='percentile')
        else:
            max_token_length = get_encoding_size(super_type_inputs, tokenizer)
            self.inputs = tokenizer(super_type_inputs, padding=True, return_tensors='pt', max_length=max_token_length, truncation=True)
        self.labels = get_super_type_labels(super_type_labels, super_type_map, multi_label=multi_label)
        self.i2c = {v: k for k, v in super_type_map.items()}
        self.multi_label = multi_label
        self.num_classes = len(super_type_map)
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return {
            'input_ids': self.inputs['input_ids'][idx],
            'attention_mask': self.inputs['attention_mask'][idx],
            'labels': self.labels[idx]
        }


class EntityClassificationDataset(Dataset):
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

class GPT2Dataset(Dataset):
    def __init__(self, tokenized):
        self.tokenized = tokenized
    
    def __len__(self):
        return len(self.tokenized['input_ids'])
    
    def __getitem__(self, index):
        item = {key: val[index] for key, val in self.tokenized.items()}
        return item
    

class LinkPredictionDataset(DGLDataset):
    def __init__(self, graphs, tokenizer, model, split_type='train', test_size=0.2, raw_dir='datasets/LP', save_dir='datasets/LP'):
        self.raw_graphs = graphs
        self.tokenizer = tokenizer
        self.model = model
        self.test_size = test_size
        self.split_type = split_type
        
        super().__init__(name='link_prediction', raw_dir=raw_dir, save_dir=save_dir)
        """
        Load dataset of graphs if exists, otherwise create it.
        """
        
        
    def __getitem__(self, idx):
        return self.graphs[idx]
    
    def __len__(self):
        return len(self.graphs)
    
    def process(self):
        self.graphs = self._prepare()

    def _prepare(self):
        prepared_graphs = [self._prepare_graph(g) for g in tqdm(self.raw_graphs, desc='Preparing graphs')]
        return prepared_graphs
    
    def _prepare_graph(self, g):

        node_strs = [promptize_node(g, n) for n in g.nodes()]
        max_token_length = get_encoding_size(node_strs, self.tokenizer)
        node_encodings = self.tokenizer(node_strs, padding=True, truncation=True, max_length=max_token_length, return_tensors='pt')
        node_embeddings = get_embedding(self.model, node_encodings).cpu()
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
            dgl.save_graphs(os.path.join(self.save_dir, f'{self.name}_{k}_{self.split_type}.dgl'), v)
    
    
    def load(self):
        """Load list of DGLGraphs using DGL load_graphs."""
        print("Loading graphs from cache...")
        
        keys = ['train_pos_g', 'train_neg_g', 'test_pos_g', 'test_neg_g', 'train_g']
        k_graphs = {k: [] for k in keys}
        for k in keys:
            k_graphs[k] = dgl.load_graphs(os.path.join(self.save_dir, f'{self.name}_{k}_{self.split_type}.dgl'))[0]
        
        self.graphs = list()
        for i in range(len(k_graphs['train_g'])):
            self.graphs.append({k: v[i] for k, v in k_graphs.items()})
        
        print(f'Loaded {len(self.graphs)} graphs.')

        
    def has_cache(self):
        print("Checking if cache exists at: ", os.path.join(self.save_dir, f'{self.name}_train_g_{self.split_type}.dgl'))
        return os.path.exists(os.path.join(self.save_dir, f'{self.name}_train_g_{self.split_type}.dgl'))
    
