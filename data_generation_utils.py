import os
import pickle
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
from models import UMLGPT, UMLGPTClassifier
from stqdm import stqdm
from constants import *


"""
    This file contains the utility functions for graph data, pytorch datasets generation and processing.
"""


"""
Constants used as special tokens to tokenize graph data
"""


"""
Lambda functions to promptize the data in a graph node i.e., given a triple (entity, relations, super_type),
where entity is the name of the node, relations are the edges from the node and super_type is the super type of the node,

the ``promptize_triple`` function returns a string with the following format:
    <SOS> <superType> super_type </superType> <entity> entity </entity> <relations> relations </relations> <EOS>

the ``promptize_node`` function returns a string promptized triple of the node:
    <SOS> <superType> super_type </superType> <entity> entity </entity> <relations> relations </relations> <EOS>

the ``promptize_super_type_generation`` function returns a string promptized triple of the super type generation task:
    <SOS> <entity> entity </entity> <relations> relations </relations> <SEP> <superType> super_type </superType> <EOS>

the ``promptize_entity_type_generation`` function returns a string promptized triple of the entity type generation task:
    <SOS> <superType> super_type </superType> <relations> relations </relations> <SEP> <entity> entity </entity> <EOS>

the ``promptize_super_type_classification`` function returns a tuple (str, super_type) promptized triple of the super type classification task:
    <SOS> <entity> entity </entity> <relations> relations </relations> <EOS>, <superType> super_type </superType>

the ``promptize_entity_type_classification`` function returns a tuple (str, entity) promptized triple of the entity type classification task:
    <SOS> <superType> super_type </superType> <relations> relations </relations> <EOS>, <entity> entity </entity>
"""

promptize_triple = lambda x: f"{SOS} {SSP} {clean_text(x[2])} {ESP} {SEN} {clean_text(x[0])} {EEN} {SRP} {clean_text(x[1])} {ERP} {EOS}"
promptize_node = lambda g, n: promptize_triple((n, g.nodes[n]['references'] if 'references' in g.nodes[n] else '', g.nodes[n]['super_types'] if 'super_types' in g.nodes[n] else ''))

promptize_super_type_generation = lambda x: f"{SOS} {SEN} {clean_text(x[0])} {EEN} {SRP} {clean_text(x[1])} {ERP} {SEP} {SSP} {clean_text(x[2])} {ESP} {EOS}"
promptize_entity_type_generation = lambda x: f"{SOS} {SSP} {clean_text(x[1])} {ESP} {SRP} {clean_text(x[1])} {ERP} {SEP} {SEN} {clean_text(x[0])} {EEN} {EOS}"

promptize_super_type_classification = lambda x: (f"{SOS} {SEN} {clean_text(x[0])} {EEN} {SRP} {clean_text(x[1])} {ERP} {EOS}", f"{clean_text(x[2])}")
promptize_entity_type_classification = lambda x: (f"{SOS} {SSP} {clean_text(x[1])} {ESP} {SRP} {clean_text(x[1])} {ERP} {EOS}", f"{clean_text(x[0])}")



def remove_duplicates(data):
    """
    remove_duplicates function removes duplicate samples from the data
    """
    return list({str(i): i for i in data}.values())


def print_sample_data(data):
    for split_type in data:
        print(f"Split type: {split_type}")
        print(f"Total number of samples: {len(data[split_type])}")
        print(f"2 Samples: {data[split_type][:2]}")
        print()


def get_promptized_data_for_super_type_generation(data):
    """
    ``get_promptized_data_for_super_type_generation`` function returns data with promptized triples for super type generation task
        i.e., given a node triple (entity, relations, super_type), it returns a string promptized triple of the super type generation task:
        [entity relations SEP super_type]
    """

    promptized_data = {
        split_type: remove_duplicates([promptize_super_type_generation(i) for i in data[split_type] if len(i[2].strip())])\
              for split_type in data
    }
    # print_sample_data(promptized_data)
    
    return promptized_data

def get_promptized_data_for_entity_generation(data):
    """
    ``get_promptized_data_for_entity_generation`` function returns data with promptized triples for entity generation task
        i.e., given a node triple (entity, relations, super_type), it returns a string promptized triple of the entity generation task:
        [super_type relations SEP entity]
    """
    promptized_data = {
        split_type: remove_duplicates(
            [promptize_entity_type_generation(i) for i in data[split_type] if len(i[1].strip())])\
              for split_type in data
    }
    # print_sample_data(promptized_data)
    return promptized_data


def get_promptized_data_for_super_type_classification(data):
    """
    ``get_promptized_data_for_super_type_classification`` function returns data with promptized triples for super type classification task
        i.e., given a node triple (entity, relations, super_type), it returns a tuple (str, super_type) promptized triple of the super type classification task:
        ([entity relations], super_type)
    """

    promptized_data = {
        split_type: remove_duplicates(
            [promptize_super_type_classification(i) for i in data[split_type] if len(i[2].strip())])\
              for split_type in data
    }
    print_sample_data(promptized_data)
    
    return promptized_data


def get_promptized_data_for_entity_classification(data):
    """
        ``get_promptized_data_for_entity_classification`` function returns data with promptized triples for entity classification task
        i.e., given a node triple (entity, relations, super_type), it returns a tuple (str, entity) promptized triple of the entity classification task:
        ([super_type relations], entity)

    """

    promptized_data = {
        split_type: remove_duplicates([promptize_entity_type_classification(i) for i in data[split_type] if len(i[1].strip())])\
              for split_type in data
    }
    print_sample_data(promptized_data)
    return promptized_data

def get_promptized_data_for_generation(data):

    """
        ``get_promptized_data_for_generation`` function returns data with promptized triples for generation task
            i.e., given a node triple (entity, relations, super_type), it returns a string promptized triple of the generation task:
            [super_type entity relations]
    """

    data_for_super_type_generation = get_promptized_data_for_super_type_generation(data)
    data_for_entity_generation = get_promptized_data_for_entity_generation(data)

    promptized_data = {
        split_type: data_for_super_type_generation[split_type] + data_for_entity_generation[split_type]\
              for split_type in data
    }
    print_sample_data(promptized_data)
    
    return promptized_data


def get_classification_dataset(data, tokenizer, encoder, class_type):
    """
    ``get_classification_dataset`` function returns data with promptized triples for classification task
    """

    if class_type == 'super_type':
        promptized_data = get_promptized_data_for_super_type_classification(data)
    elif class_type == 'entity':
        promptized_data = get_promptized_data_for_entity_classification(data)
    else:
        raise ValueError(f"Class type {class_type} not supported")

    dataset = {
        split_type: UMLNodeDataset(
            promptized_data[split_type], tokenizer, encoder) for split_type in promptized_data
    }
    return dataset


def get_super_type_labels(super_types, super_type_map, multi_label=False):
    """
    ``get_super_type_labels`` function returns super type labels for the super type classification task
        i.e., given a list of super types, it returns a list of super type labels if multi_label is False, else it returns a list of lists of super type labels
    """

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
    """
    ``get_encoding_size`` function returns the encoding size for the given data and tokenizer
        i.e., given a list of strings, it returns the 99.5th percentile of the lengths of the tokenized strings
    99.5th percentile is used to avoid the out of memory error while training the model
    """

    tokens = tokenizer(data)
    lengths = [len(i) for i in tokens['input_ids']]
    size = int(np.percentile(lengths, 99.5))
    # print("Encoding size: ", size)
    return size


def get_pretrained_lm_tokenizer(model_name, special_tokens=SPECIAL_TOKENS):
    """
        ``get_pretrained_lm_tokenizer`` function returns the tokenizer for the given hugging face language model
    """

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.add_special_tokens({'additional_special_tokens': special_tokens})
    if tokenizer.pad_token is None:
        tokenizer.pad_token = "<pad>"

    print("Vocab size: ", len(tokenizer))
    return tokenizer


def get_word_tokenizer_tokenizer(data, lower=True, special_tokens=SPECIAL_TOKENS):
    """
    ``get_word_tokenizer_tokenizer`` function constructs a custom Vocabulary tokenizer for the given data
    """

    tokenizer = VocabTokenizer(data, lower=lower, special_tokens=special_tokens)
    print("Vocab size: ", len(tokenizer))
    return tokenizer


def get_generative_uml_dataset(data, tokenizer):
    """
    ``get_generative_uml_dataset`` function returns the dataset for the generative task
    split_type: train, test, unseen
    """

    dataset = {
        split_type: GenerativeUMLDataset(data[split_type], tokenizer) for split_type in data
    }
    return dataset


def get_dataloaders(dataset, batch_size=32):
    """
    ``get_dataloaders`` function returns the dataloaders for the given dataset
    split_type: train, test, unseen
    """

    dataloaders = {
        split_type: DataLoader(
            dataset[split_type], batch_size=batch_size, shuffle=split_type == 'train') for split_type in dataset
    }
    return dataloaders


def get_gpt2_tokenized_data(data, tokenizer):
    """
    ``get_gpt2_tokenized_data`` function returns the tokenized data for the given data and GPT2 tokenizer
    This tokenizer is used for GPT2 training
    """
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
    """
    ``get_gpt2_dataset`` function returns the dataset for the GPT2 training
    split_type: train, test, unseen
    """
    tokenized_data = get_gpt2_tokenized_data(data, tokenizer)
    dataset = {
        split_type: EncodingsDataset(tokenized_data[split_type]) for split_type in data
    }
    return dataset


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


def get_kfold_lm_data(data, seed=42, test_size=0.1, phase=TRAINING_PHASE):
    """
    ``get_kfold_lm_data`` function returns a generator of k-fold data for the generative task
    seen_graph_triples are the all the triples from the graphs that are considered available for training a language model
        out of which 90% is used for training and 10% is used for validation
    
    unseen_graph_triples are the all the triples from the graphs that are considered unavailable for training a language model
        and are used for testing the language model
    """

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

        if phase == TRAINING_PHASE:
            data = {
                TRAIN_LABEL: seen_train,
                TEST_LABEL: seen_test,
                UNSEEN_LABEL: unseen_graph_triples,
            }
        else:
            data = {
                TEST_LABEL: seen_test + unseen_graph_triples
            }
        
        i += 1
        yield data


def get_kfold_lp_data(data, seed=42, test_size=0.1, phase='train'):
    """
    ``get_kfold_lp_data`` function returns a generator of k-fold data for the link prediction task
    seen_graphs are the all the graphs that are considered available for training a link prediction model
        out of which 90% is used for training and 10% is used for validation

    unseen_graphs are the all the graphs that are considered unavailable for training a link prediction model
        and are used for testing the link prediction model
    
    Each graph further has 20% of its edges masked for testing the link prediction model
    """
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

        if phase == TRAINING_PHASE:
            data = {
                TRAIN_LABEL: seen_train,
                TEST_LABEL: seen_test,
                UNSEEN_LABEL: unseen_graphs,
            }
        else:
            data = {
                TEST_LABEL: seen_test + unseen_graphs
            }
        
        i += 1
        yield data


def get_tokenization(tokenizer, data):
    """
    ``get_tokenization`` function returns the tokenization for the given tokenizer and data
    """
    if isinstance(tokenizer, VocabTokenizer):
        tokenized_data = tokenizer.batch_encode(
            data, return_tensors='pt', max_length='percentile')
    else:
        tokenized_data = tokenizer(
            data, return_tensors='pt', padding=True)
    return tokenized_data


def get_embedding(model, encodings, pooling='last'):
    """
    ``get_embedding`` function returns the embeddings for the given model and encodings
    pooling: last, mean, max, min, sum, cls
    pooling is used to pool the embeddings of the tokens in the sequence
    e.g., if pooling is last, the last token embedding is used as the embedding for the sequence
    if pooling is mean, the mean of the token embeddings is used as the embedding for the sequence
    """
    encoding_dataset = EncodingsDataset(encodings)
    encoding_dataloader = torch.utils.data.DataLoader(encoding_dataset, batch_size=128, shuffle=False)
    model.eval()

    with torch.no_grad():
        embeddings = list()
        for batch in encoding_dataloader:

            if isinstance(model, UMLGPT) or isinstance(model, UMLGPTClassifier):
                outputs = model.get_embedding(batch['input_ids'].to(DEVICE), batch['attention_mask'].to(DEVICE))
            else:
                encodings = {k: v.to(DEVICE) for k, v in batch.items()}
                outputs = model(**encodings)[0]

            outputs = outputs.cpu().detach()
            if pooling == 'last':
                outputs = outputs[:, -1, :]
            elif pooling == 'mean':
                outputs = torch.mean(outputs, dim=1)
            elif pooling == 'max':
                outputs = torch.max(outputs, dim=1)[0]
            elif pooling == 'min':
                outputs = torch.min(outputs, dim=1)[0]
            elif pooling == 'sum':
                outputs = torch.sum(outputs, dim=1)
            elif pooling == 'cls':
                outputs = outputs[:, 0, :]
            else:
                raise ValueError(f"Pooling {pooling} not supported")
            embeddings.append(outputs)
        
        embeddings = torch.cat(embeddings, dim=0)
        
    return embeddings


class VocabTokenizer:
    """
    class VocabTokenizer is a custom tokenizer that is used to tokenize the graph data
    It is used to tokenize the graph data for the generative task
    The vocabulary is constructed using the words in the graph data
    The encode, decode and batch_encode functions are used to encode, decode and batch encode the graph data
    The tokenizer takes special tokens as input and adds them to the vocabulary
    The special tokens are used to tokenize the graph data and treat them as special tokens
    """

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
        """
        ``batch_encode`` function encodes the given list of strings

        Args:
            x: list of strings to be encoded
            return_tensors: whether to return tensors or lists
            max_length: maximum length of the encoded tokens
            max_length == 'percentile': maximum length is set to 99.95th percentile of the lengths of the encoded tokens
        """

        assert isinstance(x, list), "Input must be a list"
        # batch_encodings = [self.encode(i) for i in tqdm(x, desc='Encoding')]
        batch_encodings = [self.encode(i) for i in x]
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
        """
        ``encode`` function encodes the given string
        e.g.,
            input: "hello world"
            output: [1, 2]
        """
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
        """
        ``decode`` function decodes the given list of integers
        e.g.,
            input: [1, 2]
            output: "hello world"
        Inverse of encode function
        """
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
        return f"VocabTokenizer(vsize={len(self.vocab)})"
    
    def save_pretrained(self, save_directory):
        """
        ``save_pretrained`` function saves the tokenizer to the given directory
        """
        if not os.path.exists(save_directory):
            os.makedirs(save_directory)
        tokenizer_file = os.path.join(save_directory, 'tokenizer.pkl')
        pickle.dump(self, open(tokenizer_file, 'wb'))
        print(f"Tokenizer saved to {tokenizer_file}")

    def __name__(self):
        return str(self)
    
    @property
    def name_or_path(self):
        return str(self)

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
    
