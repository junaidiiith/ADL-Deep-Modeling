import argparse
import logging
import os
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--num_epochs", type=int, default=10)
    parser.add_argument("--data_dir", type=str, default="datasets/ecore_graph_pickles")
    parser.add_argument("--graphs_file", type=str, default="combined_graphs_clean.pkl")
    parser.add_argument("--model_name", type=str, default="xlm-roberta-base")
    parser.add_argument("--alpha", type=float, default=0.5)

    parser.add_argument("--multi_label", action="store_true")

    args = parser.parse_args()
    logging.info(args)
    return args
