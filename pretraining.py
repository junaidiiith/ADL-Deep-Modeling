from parameters import parse_args
from nx2str import get_graph_data
from training_utils import train_umlgpt, train_hugging_face_gpt
from uml_data_generation import get_kfold_lm_data, get_promptized_data_for_generation
from common_utils import create_run_config
from constants import UMLGPTMODEL


"""
This file contains the code for pretraining the UML-GPT model.
The pretraining is done on the graph data.
"""

def main(args):

    create_run_config(args)
    print(args)

    # exit(0)
    graph_data = get_graph_data(args.graphs_file)
    for _, data in enumerate(get_kfold_lm_data(graph_data, seed=args.seed, phase=args.phase)):
        print("Running fold:", _)

        print("Creating dataset...")
        dataset = get_promptized_data_for_generation(data)

        print("Initializing...", dataset.keys())
        # print(dataset['test'][0])
        if args.gpt_model == UMLGPTMODEL:
            train_umlgpt(dataset, args)
        else:
            train_hugging_face_gpt(dataset, args)

        ### Comment the break statement to train on all the folds
        break

# if __name__ == '__main__':
#     args = parse_args()
#     main(args)
