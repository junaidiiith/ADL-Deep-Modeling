from parameters import parse_args
import os
import pickle
from nx2str import get_graph_data
from uml_data_generation import get_kfold_lp_data
from common_utils import create_run_config
from models import UMLGPT
from transformers import AutoModel
from training_utils import get_tokenizer
from datasets import LinkPredictionDataset
from dgl.dataloading import GraphDataLoader
import dgl
from models import GNNModel, MLPPredictor
from trainers import GNNLinkPredictionTrainer
from constants import *

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
    lp_trainer = GNNLinkPredictionTrainer(gnn_model, predictor, args)

    for split_type in graphs:
        print(f"Training Link Prediction {split_type} graphs")
        # with st.empty():
        #     st.write(f"Training Link Prediction on {split_type} graphs")

        dataset_prefix = f"{split_type}_ip={input_dim}_tok={os.path.basename(tokenizer.name_or_path)}"
        dataset = LinkPredictionDataset(
            graphs=graphs[split_type], 
            tokenizer=tokenizer, 
            model=language_model, 
            test_size=args.test_size, 
            prefix=dataset_prefix
        )
        dataloader = GraphDataLoader(
            dataset, 
            batch_size=args.batch_size, 
            shuffle=True, 
            collate_fn=collate_graphs
        )
        lp_trainer.run_epochs(dataloader, args.num_epochs)


def main(args):
    """
        This function trains the link prediction task
        It loads the graph data from the path provided in args
        It then creates the dataset for link prediction
        It then trains the link prediction task
    """
    create_run_config(args)
    graph_data = get_graph_data(args.graphs_file)
    for i, graphs in enumerate(get_kfold_lp_data(graph_data, phase=args.phase)):
        break
    
    train_link_prediction(graphs, args)


# if __name__ == '__main__':
#     args = parse_args()
#     main(args)
