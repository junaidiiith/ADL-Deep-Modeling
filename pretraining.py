from parameters import parse_args
from graph_utils import get_graph_data
from trainers import train_umlgpt, train_hugging_face_gpt
from data_generation_utils import get_kfold_lm_data, get_promptized_data_for_generation
from utils import create_run_config
from constants import UMLGPTMODEL

"""
This file contains the code for pretraining the UML-GPT model.
The pretraining is done on the graph data.
"""

def main(args):
    create_run_config(args)
    # print(config)
    # print("Loading graph data from:", args.graphs_file)

    graph_data = get_graph_data(args.graphs_file)
    for _, data in enumerate(get_kfold_lm_data(graph_data, seed=args.seed, phase=args.phase)):
        break
    
    print("Creating dataset...")
    dataset = get_promptized_data_for_generation(data)

    print("Initializing...", dataset.keys())
    # print(dataset['test'][0])
    if args.gpt_model == UMLGPTMODEL:
        train_umlgpt(dataset, args)
    else:
        train_hugging_face_gpt(dataset, args)


if __name__ == '__main__':
    args = parse_args()
    main(args)
