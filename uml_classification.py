#! Description: This file contains the code for training the UML-GPT model for classification task.

import pandas as pd
import streamlit as st
import pickle
from parameters import parse_args
from nx2str import get_graph_data
from trainers.hf_classifier import ClassificationTrainer
from training_utils import get_hf_classification_model, get_uml_gpt
from uml_data_generation import get_kfold_lm_data
from uml_data_generation import get_classification_dataset
from uml_data_generation import get_dataloaders
from models import UMLGPTClassifier
from trainers.umlgpt import UMLGPTTrainer
from metrics import get_recommendation_metrics
from training_utils import get_tokenizer
from constants import TRAIN_LABEL, TRAINING_PHASE, UMLGPTMODEL, WORD_TOKENIZER, INFERENCE_PHASE

from common_utils import create_run_config


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

    model = get_uml_gpt(vocab_size=len(tokenizer), args=args)

    dataset = get_classification_dataset(data, tokenizer, label_encoder, args.class_type)
    uml_gpt_classifier = UMLGPTClassifier(model, len(label_encoder))
    uml_gpt_trainer = UMLGPTTrainer(uml_gpt_classifier, get_dataloaders(dataset), args, compute_metrics_fn=compute_metrics_fn)
    if args.phase == TRAINING_PHASE:
        uml_gpt_trainer.train(args.num_epochs)
    else:
        results = uml_gpt_trainer.evaluate()
        st.dataframe([results], hide_index=True)

    
    if args.phase == INFERENCE_PHASE:
        inverse_label_encoder = {v: k for k, v in label_encoder.items()}
        recommendations = uml_gpt_trainer.get_recommendations()
        recommendations = {inverse_label_encoder[k]: [inverse_label_encoder[v] for v in recommendations[k]] for k in recommendations}
        df = pd.DataFrame(recommendations.items(), columns=[f'Class', 'Recommendations'])
        df.insert(0, '#', range(1, len(df)+1))
        with st.empty():
            st.write("Recommendations")
            st.dataframe(df, height=500, hide_index=True)



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
    dataset[TRAIN_LABEL].num_classes = len(label_encoder)

    model = get_hf_classification_model(
        args.from_pretrained, dataset[TRAIN_LABEL].num_classes, tokenizer)

    trainer = ClassificationTrainer(model, tokenizer, dataset, get_recommendation_metrics, args)

    if args.phase == TRAINING_PHASE:
        trainer.train(args.num_epochs)
        trainer.save_model()    
    else:
        results = trainer.evaluate()
        st.dataframe([results], hide_index=True)


    if args.phase == INFERENCE_PHASE:
        recommendations = trainer.get_recommendations()
        recommendations = {label_encoder[k]: [label_encoder[v] for v in recommendations[k]] for k in recommendations}

        with st.empty():
            st.write("Recommendations")
            for label in recommendations:
                st.write(f"{label}: {recommendations[label]}")


def main(args):
    create_run_config(args)
    # exit(0)
    # st.json(config)
    graph_data = get_graph_data(args.graphs_file)
    entity_map, super_types_map = graph_data['entities_encoder'], graph_data['super_types_encoder']
    for i, data in enumerate(get_kfold_lm_data(graph_data, seed=args.seed)):
        print("Running fold:", i)
        
        label_encoder = super_types_map if args.class_type == 'super_type' else entity_map

        if args.classification_model in [UMLGPTMODEL]:
            train_uml_gpt_classification(data, label_encoder, compute_metrics_fn=get_recommendation_metrics, args=args)
        else:
            pretrained_lm_sequence_classification(data, label_encoder, args)

        ### Comment the break statement to train on all the folds
        break

# if __name__ == '__main__':
#     args = parse_args()
#     main(args)
