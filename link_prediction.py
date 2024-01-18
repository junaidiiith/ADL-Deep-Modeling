import streamlit as st
import pandas as pd
from parameters import parse_args
import os
import pickle
from nx2str import get_graph_data
from uml_data_generation import get_kfold_lp_data
from common_utils import create_run_config
from models import UMLGPT
from transformers import AutoModel
from training_utils import get_tokenizer
from link_prediction_dataset import LinkPredictionDataset
from dgl.dataloading import GraphDataLoader
import dgl
from models import GNNModel, MLPPredictor
from trainers.link_predictor import GNNLinkPredictionTrainer
from constants import *


def describe_graph_dataloader(dataloader, split_type):
    """
        Describe the graph dataloader
        Prints - 
        1. total number of nodes
        2. average number of nodes
        3. total number of edges
        4. average number of edges
        5. total and average train positive and negative edges
        6. total and average test positive and negative edges
        
        'train_pos_g': train graph with positive edges 
        'train_neg_g': train graph negative edges
        'test_pos_g': test graph with positive edges
        'test_neg_g': test graph with negative edges
        
    """
    mean_and_median = lambda x: (sum(x), sum(x) / len(x), sorted(x)[len(x) // 2])
    nodes = [g['train_g'].num_nodes() for g in dataloader]
    train_pos_edges = [g['train_pos_g'].num_edges() for g in dataloader]
    train_neg_edges = [g['train_neg_g'].num_edges() for g in dataloader]
    test_pos_edges = [g['test_pos_g'].num_edges() for g in dataloader]
    test_neg_edges = [g['test_neg_g'].num_edges() for g in dataloader]

    d = {
        'Nodes': mean_and_median(nodes),
        'Train Pos Edges': mean_and_median(train_pos_edges),
        'Train Neg Edges': mean_and_median(train_neg_edges),
        'Test Pos Edges': mean_and_median(test_pos_edges),
        'Test Neg Edges': mean_and_median(test_neg_edges),
    }
    df = pd.DataFrame.from_dict(d, orient='index', columns=['Total', 'Average', 'Median'])
    print(df)
    with st.expander(f"{split_type} Graphs Description"):
        
        st.dataframe(df)


def collate_graphs(graphs):
    """
        Collate a list of graphs for the link prediction task
        This method is used by the GraphDataLoader
        Given a list of graphs, with each graph having five different dgl graphs
        This method collates the graphs into a single dgl graph
        Five dgl graphs in each entry of the list are:
            1. train_g
            2. train_pos_g
            3. test_pos_g
            4. train_g_neg
            5. test_g_neg
    """
    collated_graph = {k: list() for k in graphs[0].keys()}
    for g in graphs:
        for k, v in g.items():
            collated_graph[k].append(v)
    
    for k, v in collated_graph.items():
        collated_graph[k] = dgl.batch(v)
    return collated_graph


def import_model(args, tokenizer):
    """
        Import the language model for link prediction
        If the model is uml-gpt, then the pretrained model is loaded from the path provided in args
        If the model is not uml-gpt, then the model is loaded from the huggingface transformers library
    """
    
    try:
        if args.embedding_model == UMLGPTMODEL:
            assert args.from_pretrained, "Pretrained model path is required for link prediction to get node embeddings"
            print("Loading UML-GPT model from:", args.from_pretrained)
            return UMLGPT.from_pretrained(args.from_pretrained)
        
        else:
            print("Loading model from:", args.embedding_model)
            model = AutoModel.from_pretrained(args.embedding_model, ignore_mismatched_sizes=True)
            model.resize_token_embeddings(len(tokenizer))
            return model

    except Exception as e:
        print(e)
        raise Exception("Could not import model")
    

def train_link_prediction(graphs, args):
    """
        Train the link prediction task
        graphs: dictionary of graphs for link prediction
        args: arguments

        This method creates a language model, a GNN model and a predictor model
        The language model is used to get node embeddings
        The GNN model is used to get node embeddings from the graph structure
        The predictor model is used to predict the link between two nodes
    """
    
    if args.tokenizer_file is not None and args.tokenizer_file.endswith('.pkl'):
        tokenizer = pickle.load(open(args.tokenizer_file, 'rb'))
    else:
        tokenizer = get_tokenizer(args.tokenizer)
    
    language_model = import_model(args, tokenizer)

    input_dim = language_model.token_embedding_table.weight.data.shape[1] if args.embedding_model == UMLGPTMODEL else language_model.config.hidden_size
    
    args.embedding_model = language_model.name_or_path
    print("Embed model", args.embedding_model)
    # exit(0)

    if args.phase == TRAINING_PHASE:
        
        gnn_model = GNNModel(
            model_name='SAGEConv', 
            input_dim=input_dim, 
            hidden_dim=args.embed_dim,
            out_dim=args.embed_dim,
            num_layers=args.num_layers, 
            residual=True,
        )

        predictor = MLPPredictor(
            h_feats=args.embed_dim,
            num_layers=2,
        )
        # print(language_model, tokenizer, gnn_model, predictor)
    else:
        with st.spinner("Loading trained GNN Model"):
            pth = args.gnn_location
            gnn_model = GNNModel.from_pretrained(pth)
            predictor = MLPPredictor.from_pretrained(pth)

    lp_trainer = GNNLinkPredictionTrainer(gnn_model, predictor, args)
    graphs_dataset_file = args.graphs_file.split(os.sep)[-1]
    print("GDF", graphs_dataset_file)
    for split_type in graphs:
        print(f"Training Link Prediction {split_type} graphs")
        
        dataset_prefix = f"{graphs_dataset_file}/{split_type}_ip={input_dim}_tok={os.path.basename(tokenizer.name_or_path)}"
        dataset = LinkPredictionDataset(
            graphs=graphs[split_type], 
            tokenizer=tokenizer, 
            model=language_model, 
            test_size=args.test_size, 
            prefix=dataset_prefix
        )
        print("Dataset of size: ", len(dataset))
        dataloader = GraphDataLoader(
            dataset, 
            batch_size=args.batch_size, 
            shuffle=True, 
            collate_fn=collate_graphs
        )
        
        describe_graph_dataloader(dataloader, split_type)
        
        if args.phase == TRAINING_PHASE:
            print("Training link prediction")
            print("Dataset Description")

            results = lp_trainer.run_epochs(dataloader, args.num_epochs)
            print(results)
            lp_trainer.save_model()
            
        else:
            loss, accuracy = lp_trainer.test(dataloader)

            with lp_trainer.st_results.container():
                st.markdown(f"## Results for {split_type} graphs")
                st.markdown(f"### Loss: {loss:.3f}")
                st.markdown(f"### Accuracy: {accuracy:.3f}")

        print(f"Results for {split_type} graphs")
        

def main(args):
    """
        This function trains the link prediction task
        It loads the graph data from the path provided in args
        It then creates the dataset for link prediction
        It then trains the link prediction task
    """
    create_run_config(args)
    # exit(0)

    graph_data = get_graph_data(args.graphs_file)
    for i, graphs in enumerate(get_kfold_lp_data(graph_data, phase=args.phase)):
        
        print("Running fold:", i)
        train_link_prediction(graphs, args)
        ### Comment the break statement to train on all the folds
        break

if __name__ == '__main__':
    args = parse_args()
    main(args)
