import os
from parameters import parse_args
from graph_utils import get_graph_data
from trainers import train_umlgpt, train_hugging_face_gpt
from data_generation_utils import get_kfold_lm_data, get_promptized_data_for_generation
from utils import create_run_config


"""
This file contains the code for pretraining the UML-GPT model.
The pretraining is done on the graph data.
"""


if __name__ == '__main__':
    args = parse_args()
    args.stage = 'pre'
    config = create_run_config(args)
    print(config)


    data_dir = args.data_dir

    graph_data = get_graph_data(os.path.join(data_dir, args.graphs_file))
    label_map, super_type_map = graph_data['entities_encoder'], graph_data['super_types_encoder']
    inverse_label_map = {v: k for k, v in label_map.items()}
    inverse_super_type_map = {v: k for k, v in super_type_map.items()}

    label_map, super_type_map = graph_data['entities_encoder'], graph_data['super_types_encoder']
    for i, data in enumerate(get_kfold_lm_data(graph_data, seed=args.seed)):
        break
    
    print("Creating dataset...")
    dataset = get_promptized_data_for_generation(data)

    print("Initializing...")

    if args.gpt_model == 'uml-gpt':
        train_umlgpt(dataset, args)
    else:
        train_hugging_face_gpt(dataset, args)
