import os
from tqdm.auto import tqdm
from parameters import parse_args
from graph_utils import get_graph_data
from data_generation_utils import get_kfold_lp_data
from utils import create_run_config
from models import UMLGPT
from transformers import AutoModel
from trainers import get_tokenizer
from data_generation_utils import LinkPredictionDataset
from dgl.dataloading import GraphDataLoader
import dgl
from models import GNNModel, MLPPredictor
from trainers import GNNLinkPredictionTrainer


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


def import_model(args):
    """
        Import the language model for link prediction
        If the model is uml-gpt, then the pretrained model is loaded from the path provided in args
        If the model is not uml-gpt, then the model is loaded from the huggingface transformers library
    """
    try:
        if args.gpt_model == 'uml-gpt':
            assert args.from_pretrained, "Pretrained model path is required for link prediction to get node embeddings"
            return UMLGPT.from_pretrained(args.from_pretrained)
        else:
            if args.from_pretrained:
                return AutoModel.from_pretrained(args.from_pretrained)
            else:
                AutoModel.from_pretrained(args.gpt_model)

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
    language_model = import_model(args)
    tokenizer = get_tokenizer(args.tokenizer)
    input_dim = language_model.token_embedding_table.weight.data.shape[1]

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

    lp_trainer = GNNLinkPredictionTrainer(gnn_model, predictor, args)

    for split_type in graphs:
        print(f"Training Link Prediction {split_type} graphs")
        
        dataset = LinkPredictionDataset(
            graphs=graphs[split_type], 
            tokenizer=tokenizer, 
            model=language_model, 
            test_size=args.test_size, 
            split_type=split_type
        )
        dataloader = GraphDataLoader(
            dataset, 
            batch_size=args.batch_size, 
            shuffle=True, 
            collate_fn=collate_graphs
        )
        lp_trainer.run_epochs(dataloader, args.num_epochs)



if __name__ == '__main__':
    
    args = parse_args()
    args.stage = 'lp'
    create_run_config(args)
    data_dir = args.data_dir
    
    graph_data = get_graph_data(os.path.join(data_dir, args.graphs_file))
    label_map, super_type_map = graph_data['entities_encoder'], graph_data['super_types_encoder']
    inverse_label_map = {v: k for k, v in label_map.items()}
    inverse_super_type_map = {v: k for k, v in super_type_map.items()}

    label_map, super_type_map = graph_data['entities_encoder'], graph_data['super_types_encoder']
    for i, graphs in enumerate(get_kfold_lp_data(graph_data)):
        break

    
    train_link_prediction(graphs, args)
