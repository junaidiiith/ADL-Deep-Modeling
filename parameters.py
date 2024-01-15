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

    parser.add_argument("--stage", type=str)


    parser.add_argument("--batch_size", type=int)
    parser.add_argument("--num_epochs", type=int)
    parser.add_argument("--lr", type=float)
    parser.add_argument("--warmup_steps", type=int)
    parser.add_argument("--weight_decay", type=float)
    parser.add_argument("--data_dir", type=str, default="uploaded_data")
    parser.add_argument("--log_dir", type=str, default="logs")
    parser.add_argument("--graphs_file", type=str)
    parser.add_argument("--classification_model", type=str)
    parser.add_argument("--embedding_model", type=str)
    parser.add_argument("--tokenizer", type=str)
    parser.add_argument("--from_pretrained", type=str)
    parser.add_argument("--tokenizer_file", type=str)

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--test_size", type=float)
    parser.add_argument("--ontouml_mask_prob", type=float)
    parser.add_argument("--exclude_limit", type=int)
    parser.add_argument("--distance", type=int)
    
    
    # parser.add_argument("--from_pretrained", type=str, default=None)
    parser.add_argument("--models_dir", type=str, default="models")

    parser.add_argument("--class_type", type=str)
    parser.add_argument("--phase", type=str)


    parser.add_argument("--gpt_model", type=str)
    parser.add_argument("--embed_dim", type=int)
    parser.add_argument("--num_layers", type=int)
    parser.add_argument("--num_heads", type=int)
    parser.add_argument("--block_size", type=int)
    parser.add_argument("--pooling", type=str)


    
    
    args = parser.parse_args()
    logging.info(args)
    return args
