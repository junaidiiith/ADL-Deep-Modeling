import argparse
import logging
import os
from constants import *
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"

"""
This file defines the parameters for three stages of the pipeline:
1. Pretraining
2. Fine-tuning
3. Link Prediction

The parameters are defined in the parse_args() method.

batch_size: Batch size for training
num_epochs: Number of epochs for training
lr: Learning rate for training
warmup_steps: Number of warmup steps for training
data_dir: Directory where the graph data is stored
log_dir: Directory where the logs are stored
graphs_file: Name of the file where the graph data is stored

classification_model: Name of the classification model to use.
    Choices: ['uml-gpt', 'bert-base-cased']
    uml-gpt: UML-GPT model
    bert-base-cased: BERT model

tokenizer: Name of the tokenizer to use. 
This tokenizer can be a pretrained tokenizer or a custom tokenizer.
A custom tokenizer is built using the graph data.

seed: Seed for reproducibility
test_size: Test size for train-test split

from_pretrained: Path to the pretrained model to use for fine-tuning or link prediction
models_dir: Directory where the models are stored
"""

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--num_epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--warmup_steps", type=int, default=100)
    parser.add_argument("--data_dir", type=str, default="uploaded_data")
    parser.add_argument("--log_dir", type=str, default="logs")
    parser.add_argument("--graphs_file", type=str, default="combined_graphs_clean.pkl")
    parser.add_argument("--classification_model", type=str, default="bert-base-cased", choices=[UMLGPTMODEL, 'bert-base-cased'])
    parser.add_argument("--tokenizer", type=str, default="bert-base-cased")

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--test_size", type=float, default=0.2)
    parser.add_argument("--ontouml_mask_prob", type=float, default=0.2)
    parser.add_argument("--exclude_limit", type=int, default=100)
    parser.add_argument("--distance", type=int, default=2)
    
    
    parser.add_argument("--from_pretrained", type=str, default=None)
    parser.add_argument("--models_dir", type=str, default="models")
    

    parser.add_argument("--class_type", type=str, choices=['super_type', 'entity'], default=None)


    parser.add_argument("--gpt_model", type=str, choices=[UMLGPTMODEL, 'gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl'], default='uml-gpt')
    parser.add_argument("--embed_dim", type=int, default=128)
    parser.add_argument("--num_layers", type=int, default=6)
    parser.add_argument("--num_heads", type=int, default=8)
    parser.add_argument("--block_size", type=int, default=512)
    parser.add_argument("--pooling", type=str, default='last', choices=['mean', 'max', 'cls', 'sum', 'last'])
    
    

    args = parser.parse_args()
    logging.info(args)
    return args
