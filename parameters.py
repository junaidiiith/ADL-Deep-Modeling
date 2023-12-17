import argparse
import logging
import os
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--num_epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--warmup_steps", type=int, default=100)
    parser.add_argument("--data_dir", type=str, default="datasets/ecore_graph_pickles")
    parser.add_argument("--log_dir", type=str, default="logs")
    parser.add_argument("--graphs_file", type=str, default="combined_graphs_clean.pkl")
    parser.add_argument("--model_name", type=str, default="gpt2")
    parser.add_argument("--alpha", type=float, default=0.5)
    
    parser.add_argument("--from_pretrained", type=str, default=None)
    parser.add_argument("--models_dir", type=str, default="models")
    

    parser.add_argument("--trainer", type=str, choices=['PT', 'CT', 'HFGPT'], default='HFGPT')
    parser.add_argument("--class_type", type=str, choices=['super', 'entity'], default='super')

    parser.add_argument("--multi_label", action="store_true")

    parser.add_argument("--gpt_model", type=str, default="gpt2")
    parser.add_argument("--embed_dim", type=int, default=128)
    parser.add_argument("--num_layers", type=int, default=4)
    parser.add_argument("--num_heads", type=int, default=8)
    parser.add_argument("--block_size", type=int, default=512)

    parser.add_argument("--pretrained", action="store_true")
    
    

    args = parser.parse_args()
    logging.info(args)
    return args
