import os
from parameters import parse_args
from graph_utils import get_graph_data
from trainers import train_umlgpt, train_hugging_face_gpt
from data_generation_utils import get_kfold_data, get_promptized_data_for_generation
from data_generation_utils import SPECIAL_TOKENS
from utils import create_run_config


if __name__ == '__main__':
    args = parse_args()
    config = create_run_config(args)
    print(config)


    data_dir = args.data_dir
    args.graphs_file = os.path.join(data_dir, args.graphs_file)


    graph_data = get_graph_data(args.graphs_file)
    label_map, super_type_map = graph_data['entities_encoder'], graph_data['super_types_encoder']
    inverse_label_map = {v: k for k, v in label_map.items()}
    inverse_super_type_map = {v: k for k, v in super_type_map.items()}

    label_map, super_type_map = graph_data['entities_encoder'], graph_data['super_types_encoder']
    for i, data in enumerate(get_kfold_data(graph_data)):
        break
    
    print("Creating dataset...")
    args.special_tokens = SPECIAL_TOKENS
    dataset = get_promptized_data_for_generation(data)

    print("Initializing...")

    if args.trainer in ['PT', 'CT']:
        train_umlgpt(dataset, args)
    elif args.trainer == 'HFGPT':
        train_hugging_face_gpt(dataset, args)
