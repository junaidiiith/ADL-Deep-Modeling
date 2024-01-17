from uml_datasets import \
    UMLNodeDataset, \
    EncodingsDataset, \
    GenerativeUMLDataset
    
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import DataLoader
import torch
import numpy as np
from constants import *
from common_utils import clean_text
from tokenization import get_encoding_size


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
            promptized_data[split_type], tokenizer, encoder) for split_type in promptized_data\
            if len(promptized_data[split_type]) > 0
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


def get_kfold_lp_data(data, seed=42, test_size=0.2, phase='training'):
    """
    ``get_kfold_lp_data`` function returns a generator of k-fold data for the link prediction task
    seen_graphs are the all the graphs that are considered available for training a link prediction model
        out of which 80% is used for training and 20% is used for validation

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
