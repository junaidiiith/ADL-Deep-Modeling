import os
from tqdm.auto import tqdm
from parameters import parse_args
from graph_utils import get_graph_data
from data_generation_utils import get_kfold_lp_data
from utils import create_run_config
from models import UMLGPT
from trainers import get_tokenizer
from data_generation_utils import LinkPredictionDataset
from dgl.dataloading import GraphDataLoader
import dgl
from models import GNNModel, MLPPredictor
from trainers import GNNLinkPredictionTrainer


def collate_graphs(graphs):
    collated_graph = {k: list() for k in graphs[0].keys()}
    for g in graphs:
        for k, v in g.items():
            collated_graph[k].append(v)
    
    for k, v in collated_graph.items():
        collated_graph[k] = dgl.batch(v)
    return collated_graph



def train_link_prediction(data, args):
    assert args.from_pretrained, "Pretrained model path is required for link prediction to get node embeddings"
    model = UMLGPT.from_pretrained(args.from_pretrained)
    
    tokenizer = get_tokenizer(args.tokenizer)
    dataset = LinkPredictionDataset(data, tokenizer, model)
    dataloader = GraphDataLoader(dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_graphs)

    input_dim = model.token_embedding_table.weight.data.shape[1]

    gnn_model = GNNModel(
        model_name='SAGEConv', 
        input_dim=input_dim, 
        hidden_dim=args.embed_dim,
        out_dim=args.embed_dim,
        num_layers=args.num_layers, 
        residual=True,
    )

    predictor = MLPPredictor(
        h_feats=256,
        num_layers=2,
    )

    lp_trainer = GNNLinkPredictionTrainer(gnn_model, predictor, args)
    lp_trainer.run_epochs(dataloader, args.num_epochs)



if __name__ == '__main__':
    
    args = parse_args()
    
    create_run_config(args)
    data_dir = args.data_dir
    args.graphs_file = os.path.join(data_dir, args.graphs_file)


    graph_data = get_graph_data(args.graphs_file)
    label_map, super_type_map = graph_data['entities_encoder'], graph_data['super_types_encoder']
    inverse_label_map = {v: k for k, v in label_map.items()}
    inverse_super_type_map = {v: k for k, v in super_type_map.items()}

    label_map, super_type_map = graph_data['entities_encoder'], graph_data['super_types_encoder']
    for i, data in enumerate(get_kfold_lp_data(graph_data)):
        break

    for split_type in data:
        print("Training GNN on split:", split_type, " graphs:", len(data[split_type]))
        train_link_prediction(data[split_type][:4], args)
