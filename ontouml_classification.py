import os

from parameters import parse_args
from ontouml_data_utils import get_graphs_data_kfold, get_triples, get_triples_dataset
from trainers import get_uml_gpt
from data_generation_utils import get_encoding_size
from data_generation_utils import get_dataloaders
from models import UMLGPTClassifier
from trainers import UMLGPTTrainer
from utils import get_recommendation_metrics
from trainers import get_tokenizer
from trainers import train_hf_for_classification

from utils import create_run_config


def train_ontouml_gpt_classification(data, label_encoder, compute_metrics_fn, args):
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


    tokenizer = get_tokenizer(args.tokenizer)
    dataset = {split_type: get_triples_dataset(data[split_type], label_encoder, tokenizer) for split_type in data}
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
    dataset = {split_type: get_triples_dataset(data[split_type], label_encoder, tokenizer) for split_type in data}
    dataset['train'].num_classes = len(label_encoder)
    train_hf_for_classification(dataset, tokenizer, args)



if __name__ == '__main__':
    args = parse_args()
    args.stage = 'ontouml_cls'
    config = create_run_config(args)
    print(config)
    
    for i, (seen_graphs, unseen_graphs, label_encoder) in enumerate(get_graphs_data_kfold(args)):
        print(len(seen_graphs), len(unseen_graphs), len(label_encoder))
        train_triples_seen = get_triples(seen_graphs, distance=args.distance, train=True)
        test_triples_seen = get_triples(seen_graphs, distance=args.distance, train=False)

        train_triples_unseen = get_triples(unseen_graphs, distance=args.distance, train=True)
        test_triples_unseen = get_triples(unseen_graphs, distance=args.distance, train=False)
        data = {
            'train': train_triples_seen,
            'test': test_triples_seen,
            'unseen': test_triples_unseen,
        }

        if args.classification_model in ['uml-gpt']:
            train_ontouml_gpt_classification(data, label_encoder, compute_metrics_fn=get_recommendation_metrics, args=args)
        else:
            pretrained_lm_sequence_classification(data, label_encoder, args)