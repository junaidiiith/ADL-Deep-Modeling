#! Description: This file contains the code for training the UML-GPT model for classification task.

import json
import pandas as pd
import streamlit as st
import pickle

import torch
from parameters import parse_args
from nx2str import get_graph_data
from trainers.hf_classifier import HFClassificationTrainer
from training_utils import get_hf_classification_model
from uml_data_generation import get_kfold_lm_data
from uml_data_generation import get_classification_dataset
from uml_data_generation import get_dataloaders
from models import UMLGPT, UMLGPTClassifier
from trainers.umlgpt import UMLGPTTrainer
from metrics import get_recommendation_metrics
from training_utils import get_tokenizer
from constants import TEST_LABEL, UML_CLASSIFICATION, \
    TRAINING_PHASE, UMLGPTMODEL, WORD_TOKENIZER, DEVICE

from common_utils import create_run_config


def get_recommendations(trainer, label_encoder):
    recommendations = trainer.get_recommendations()
    inv_label_encoder = {v: k for k, v in label_encoder.items()}
    recommendations = {
        inv_label_encoder[k]: \
            [inv_label_encoder[v] for v in recommendations[k]\
                if inv_label_encoder[v] != inv_label_encoder[k]] for k in recommendations
    }

    with st.empty().container():
        st.markdown("### Recommendations")
        for label in recommendations:
            st.write(f"{label} \t>>\t {recommendations[label]}")


def get_uml_gpt_classifier(vocab_size, init_classifier, num_classes, args):
    """
        Get the UMLGPT model
        Args:
            input_dim: int
                The input dimension of the model
            args: Namespace
                The arguments
    """

    if not init_classifier:
        assert args.from_pretrained is not None, "Cannot initialize classifier from pretrained model"
        classifier = UMLGPTClassifier.from_pretrained(
            args.from_pretrained,
            num_classes=None,
            init_classifier=False
        )
        print(f'Loaded pretrained UMLGPTClassifier model from {args.from_pretrained}')
    else:
        if args.from_pretrained is None:
            uml_gpt = UMLGPT(
                vocab_size=vocab_size, 
                embed_dim=args.embed_dim, 
                block_size=args.block_size,
                n_layer=args.num_layers, 
                n_head=args.num_heads
            )
            classifier = UMLGPTClassifier(uml_gpt, num_classes=num_classes)
            print("Created UMLGPTClassifier model")
        else:
            uml_gpt = UMLGPT.from_pretrained(args.from_pretrained)
            print(f'Loaded pretrained UMLGPT model from {args.from_pretrained}')
            classifier = UMLGPTClassifier(uml_gpt, num_classes=num_classes)
        
    classifier.to(DEVICE)
    return classifier


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

    if args.tokenizer_file is not None:
        tokenizer = pickle.load(open(args.tokenizer_file, 'rb'))
    
    elif args.tokenizer == WORD_TOKENIZER:
        tokenizer = get_tokenizer(WORD_TOKENIZER, data)
    else:
        tokenizer = get_tokenizer(args.tokenizer)

    init_classifier = not (args.from_pretrained is not None and \
                           UML_CLASSIFICATION in args.from_pretrained)
    print("Initializing classifier: ", init_classifier)
    print("from pretrained: ", args.from_pretrained)
    try:
        model = get_uml_gpt_classifier(
            vocab_size=len(tokenizer), 
            init_classifier=init_classifier,
            num_classes=len(label_encoder), 
            args=args
        )
    except Exception as e:
        print("Error in creating UMLGPTClassifier model")
        print(e)
        exit(0)

    print(data.keys())
    dataset = get_classification_dataset(data, tokenizer, label_encoder, args.class_type)
    for k, v in dataset.items():
        print(k, len(v))

    if len(dataset) == 0:
        with st.empty():
            st.markdown(f"No nodes with {args.class_type} found")
        exit(0)

    dataloaders = get_dataloaders(dataset, batch_size=args.batch_size)
    uml_gpt_trainer = UMLGPTTrainer(
        model, 
        tokenizer, 
        dataloaders, 
        args, 
        compute_metrics_fn=compute_metrics_fn
    )
    
    if args.phase == TRAINING_PHASE:
        uml_gpt_trainer.train(args.num_epochs)
    else:
        results = uml_gpt_trainer.evaluate()
        print(results)
        st.dataframe([results], hide_index=True)
        get_recommendations(uml_gpt_trainer, label_encoder)


def pretrained_lm_sequence_classification(data, label_encoder, args):

    """
    This function trains the Huggingface pretrained language model for sequence classification.

    Args:
        data (dict): The graph data for the classification task with train, test, unseen graphs
        label_encoder (dict): The label encoder
        compute_metrics_fn (function): The function to compute the metrics
        args (Namespace): The arguments passed to the script

    """
    tokenizer = get_tokenizer(args.from_pretrained)
    dataset = get_classification_dataset(data, tokenizer, label_encoder, args.class_type)
    dataset[TEST_LABEL].num_classes = len(label_encoder)

    model = get_hf_classification_model(
        args.from_pretrained, dataset[TEST_LABEL].num_classes, tokenizer)

    dataloaders = {
        split_type: torch.utils.data.DataLoader(
            dataset[split_type], 
            batch_size=args.batch_size, 
            shuffle=args.phase == TRAINING_PHASE,
        ) for split_type in dataset
    }
    hf_trainer = HFClassificationTrainer(model, tokenizer, dataloaders, get_recommendation_metrics, args)

    if args.phase == TRAINING_PHASE:
        print("Training")
        hf_trainer.train(args.num_epochs)
        hf_trainer.save_model()
    else:
        print("Inference")
        results = hf_trainer.evaluate()
        print(results)
        st.markdown("## Metrics")
        st.dataframe([results], hide_index=True)


        get_recommendations(hf_trainer, label_encoder)


def main(args):
    create_run_config(args)
    graph_data = get_graph_data(args.graphs_file)
    
    super_type_encoder, entities_encoder = graph_data['super_types_encoder'], graph_data['entities_encoder']
    label_encoder = super_type_encoder if args.class_type == 'super_type' else entities_encoder

    for i, data in enumerate(get_kfold_lm_data(graph_data, seed=args.seed, phase=args.phase)):
        print("Running fold:", i)
        
        # label_encoder = json.load(open(f"{UPLOADED_DATA_DIR}/{args.class_type}_encoder.json"))
        print("Label encoder num classes: ", len(label_encoder))
        if args.classification_model == UMLGPTMODEL:
            train_uml_gpt_classification(data, label_encoder, compute_metrics_fn=get_recommendation_metrics, args=args)
        else:
            pretrained_lm_sequence_classification(data, label_encoder, args)

        ## Comment the break statement to train on all the folds
        break

if __name__ == '__main__':
    args = parse_args()
    main(args)
