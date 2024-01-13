from parameters import parse_args
from ontouml_data_utils import get_graphs_data_kfold, get_triples, get_triples_dataset
from trainers import get_tokenizer
from trainers import train_hf_for_classification

from utils import create_run_config


def pretrained_lm_sequence_classification(data, label_encoder, args):

    """
    This function trains the Huggingface pretrained language model for sequence classification.

    Args:
        data (dict): The graph data for the classification task with train, test, unseen graphs
        label_encoder (dict): The label encoder
        compute_metrics_fn (function): The function to compute the metrics
        args (Namespace): The arguments passed to the script

    """

    tokenizer = get_tokenizer(args.from_pretrained, special_tokens=[])
    dataset = {split_type: get_triples_dataset(data[split_type], label_encoder, tokenizer) for split_type in data}
    dataset['train'].num_classes = len(label_encoder)
    train_hf_for_classification(dataset, tokenizer, args)


def main(args):
    create_run_config(args)
    for i, (seen_graphs, unseen_graphs, label_encoder) in enumerate(get_graphs_data_kfold(args)):
        print(len(seen_graphs), len(unseen_graphs), len(label_encoder))
        train_triples_seen = get_triples(seen_graphs, distance=args.distance, train=True)
        test_triples_seen = get_triples(seen_graphs, distance=args.distance, train=False)

        test_triples_unseen = get_triples(unseen_graphs, distance=args.distance, train=False)
        data = {
            'train': train_triples_seen,
            'test': test_triples_seen,
            'unseen': test_triples_unseen,
        }

        pretrained_lm_sequence_classification(data, label_encoder, args)
        break


if __name__ == '__main__':
    args = parse_args()
    main(args)
