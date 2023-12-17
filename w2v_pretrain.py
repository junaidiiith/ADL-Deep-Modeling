"""
This file contains the code for pretraining the word2vec skip-gram language model on the node strings
"""

import os
import pickle
import numpy as np
from tqdm import tqdm
from gensim.models import Word2Vec
from parameters import parse_args


from graph_utils import get_graph_data
from data_utils import get_dataset

def evaluate_model(model, node_strings):
    pass


def train_w2v_model(node_strings, window_size=5, vector_size=128, epochs=50, seed=42):
    model = Word2Vec(
        node_strings,
        window=window_size,
        vector_size=vector_size,
        min_count=1,
        workers=4,
        sg=1,
        seed=seed,
        epochs=epochs
    )
    return model

def main():
    args = parse_args()
    data_dir = args.data_dir
    graphs_file = os.path.join(data_dir, args.graphs_file)
    data = get_graph_data(graphs_file)
    

    for i, graph_data in enumerate(get_dataset(data)):
    
        train_w2v_strings = [f'{i[0]} {i[1]} {i[2]}' for i in graph_data['train']]
        w2v_model = train_w2v_model(train_w2v_strings)
        w2v_model.save(os.path.join(data_dir, f'models/w2v_model_{i}.bin'))

        
