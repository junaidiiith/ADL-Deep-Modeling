import os

from parameters import parse_args
from graph_utils import get_graph_data
from trainers import get_uml_gpt
from data_generation_utils import get_kfold_lm_data
from data_generation_utils import get_data_for_classification, get_classification_dataset
from data_generation_utils import get_dataloaders
from models import UMLGPTClassifier
from trainers import UMLGPTTrainer
from utils import get_recommendation_metrics, get_recommendation_metrics_multi_label
from trainers import get_tokenizer
from trainers import train_hf_for_classification

from data_generation_utils import SPECIAL_TOKENS
from utils import create_run_config


def train_uml_gpt_classification(data, label_encoder, compute_metrics_fn, args):
    tokenizer = get_tokenizer(data, args)
    data = get_data_for_classification(data, class_type=args.class_type)
    dataset = get_classification_dataset(data, tokenizer, label_encoder, class_type=args.class_type, multi_label=args.multi_label)
    model = get_uml_gpt(len(tokenizer), args)
    uml_gpt_classifier = UMLGPTClassifier(model, len(label_encoder))
    uml_gpt_trainer = UMLGPTTrainer(uml_gpt_classifier, get_dataloaders(dataset), args, compute_metrics_fn=compute_metrics_fn)
    uml_gpt_trainer.train(args.num_epochs)


def pretrained_lm_sequence_classification(data, args):
    assert not args.multi_label, "Multi-label classification is not supported for pretrained models"
    tokenizer = get_tokenizer(data, args)
    data = get_data_for_classification(data, class_type=args.class_type)
    dataset = get_classification_dataset(data, tokenizer, label_encoder, class_type=args.class_type)
    train_hf_for_classification(dataset, tokenizer, args)



if __name__ == '__main__':
    args = parse_args()
    args.stage = 'cls'
    config = create_run_config(args)

    args.special_tokens = SPECIAL_TOKENS
    
    data_dir = args.data_dir
    args.graphs_file = os.path.join(data_dir, args.graphs_file)


    graph_data = get_graph_data(args.graphs_file)
    entity_map, super_types_map = graph_data['entities_encoder'], graph_data['super_types_encoder']
    for i, data in enumerate(get_kfold_lm_data(graph_data, seed=args.seed)):
        break
    
    compute_metrics_fn = get_recommendation_metrics_multi_label if args.multi_label else get_recommendation_metrics
    label_encoder = super_types_map if args.class_type == 'super' else entity_map

    if args.trainer in ['PT', 'CT']:
        train_uml_gpt_classification(data, label_encoder, compute_metrics_fn, args)
    elif args.trainer == 'HFGPT':
        pretrained_lm_sequence_classification(data, args)