import streamlit as st
import pickle

from graph_utils import get_graph_data
from trainers import get_uml_gpt
from data_generation_utils import get_kfold_lm_data
from data_generation_utils import get_classification_dataset
from data_generation_utils import get_dataloaders
from models import UMLGPTClassifier
from trainers import UMLGPTTrainer
from utils import get_recommendation_metrics
from trainers import get_tokenizer
from trainers import train_hf_for_classification
from constants import UMLGPTMODEL

from utils import create_run_config


def train_uml_gpt_classification(data, label_encoder, compute_metrics_fn, args):
    """
    This function trains the UML-GPT model for classification.

    This function - 
        1. creates the dataset
        2. creates a UMLGPT model using ``args.from_pretrained`` .pth file or from scratch
        3. creates a UMLGPTClassifier using the UMLGPT model
        4. creates a UMLGPTTrainer using the UMLGPTClassifier
        5. trains the UMLGPTTrainer for ``args.num_epochs`` epochs

    Args:
        data (dict): The graph data for the classification task with train, test, unseen graphs
        label_encoder (dict): The label encoder
        compute_metrics_fn (function): The function to compute the metrics
        args (Namespace): The arguments passed to the script
    """

    if args.from_pretrained is None:
        tokenizer = get_tokenizer(args.tokenizer) if args.tokenizer != 'word' else get_tokenizer('word', data)
    elif args.from_pretrained.endswith('.pth') or args.from_pretrained.endswith('.pt'):
        if args.tokenizer_file.endswith('.pkl'):
            tokenizer = pickle.load(open(args.tokenizer_file, 'rb'))
        else:
            tokenizer = get_tokenizer(args.tokenizer)
    

    dataset = get_classification_dataset(data, tokenizer, label_encoder, args.class_type)
    model = get_uml_gpt(vocab_size=len(tokenizer), args=args)
    uml_gpt_classifier = UMLGPTClassifier(model, len(label_encoder))
    uml_gpt_trainer = UMLGPTTrainer(uml_gpt_classifier, get_dataloaders(dataset), args, compute_metrics_fn=compute_metrics_fn)
    uml_gpt_trainer.train(args.num_epochs)


def pretrained_lm_sequence_classification(data, label_encoder, args):

    """
    This function trains the Huggingface pretrained language model for sequence classification.

    Args:
        data (dict): The graph data for the classification task with train, test, unseen graphs
        label_encoder (dict): The label encoder
        compute_metrics_fn (function): The function to compute the metrics
        args (Namespace): The arguments passed to the script

    """
    tokenizer = get_tokenizer(args.tokenizer)
    dataset = get_classification_dataset(data, tokenizer, label_encoder, args.class_type)
    train_hf_for_classification(dataset, tokenizer, args)


def uml_classification(args):
    config = create_run_config(args)
    st.json(config)
    graph_data = get_graph_data(args.graphs_file)
    entity_map, super_types_map = graph_data['entities_encoder'], graph_data['super_types_encoder']
    for i, data in enumerate(get_kfold_lm_data(graph_data, seed=args.seed)):
        break
    
    label_encoder = super_types_map if args.class_type == 'super_type' else entity_map

    if args.classification_model in [UMLGPTMODEL]:
        train_uml_gpt_classification(data, label_encoder, compute_metrics_fn=get_recommendation_metrics, args=args)
    else:
        pretrained_lm_sequence_classification(data, label_encoder, args)


# if __name__ == '__main__':
#     args = parse_args()
#     args.stage = 'cls'
#     config = create_run_config(args)
#     print(config)
#     uml_classification(args)
