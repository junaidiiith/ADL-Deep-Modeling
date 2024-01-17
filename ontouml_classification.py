import json
import os
import pandas as pd
import streamlit as st
import torch
from parameters import parse_args
from ontouml_data_generation import get_graphs_data_kfold, get_triples, get_triples_dataset
from training_utils import get_hf_classification_model
from trainers.hf_classifier import HFClassificationTrainer
from training_utils import get_tokenizer
from metrics import get_recommendation_metrics
from common_utils import create_run_config
from constants import TRAINING_PHASE, UPLOADED_DATA_DIR

def pretrained_lm_sequence_classification(data, args):

    """
    This function trains the Huggingface pretrained language model for sequence classification.

    Args:
        data (dict): The graph data for the classification task with train, test, unseen graphs
        label_encoder (dict): The label encoder
        compute_metrics_fn (function): The function to compute the metrics
        args (Namespace): The arguments passed to the script

    """
    label_encoder = json.load(open(
        os.path.join(UPLOADED_DATA_DIR, f'label_encoder_{args.exclude_limit}.json'), 'r'))

    tokenizer = get_tokenizer(args.tokenizer, special_tokens=[])
    dataset = {split_type: get_triples_dataset(data[split_type], label_encoder, tokenizer) for split_type in data}
    
        
    model = get_hf_classification_model(args.from_pretrained, len(label_encoder), tokenizer)
    dataloaders = {
        split_type: torch.utils.data.DataLoader(
            dataset[split_type], 
            batch_size=args.batch_size, 
            shuffle=args.phase == TRAINING_PHASE,
        ) for split_type in dataset
    }
    trainer = HFClassificationTrainer(model, tokenizer, dataloaders, get_recommendation_metrics, args)

    if args.phase == TRAINING_PHASE:
        trainer.train(args.num_epochs)
        trainer.save_model()    
    else:
        results = trainer.evaluate()
        st.dataframe([results], hide_index=True)
        results = json.load(open('results/ontouml_small.json'))
        print(results)

        inverse_label_encoder = {v: k for k, v in label_encoder.items()}
        recommendations = trainer.get_recommendations()
        recommendations = json.load(open('results/recommendations.json'))
        recommendations = {inverse_label_encoder[int(k)]: [inverse_label_encoder[int(v)] for v in recommendations[k]] for k in recommendations}
        df = pd.DataFrame(recommendations.items(), columns=[f'Class', 'Recommendations'])
        df.insert(0, '#', range(1, len(df)+1))
        with st.empty().container():
            st.write("Recommendations")
            st.dataframe(df, height=500, hide_index=True)


def main(args):
    create_run_config(args)
    # exit(0)

    for i, (seen_graphs, unseen_graphs, label_encoder) in enumerate(get_graphs_data_kfold(args)):
        le_path = os.path.join(UPLOADED_DATA_DIR, f'label_encoder_{args.exclude_limit}.json')
        
        if args.phase == TRAINING_PHASE and \
            not os.path.exists(le_path):
            json.dump(label_encoder, open(le_path, 'w'))

        print("Running fold:", i)
        print(len(seen_graphs), len(unseen_graphs))
        train_triples_seen = get_triples(seen_graphs, distance=args.distance, train=True)
        test_triples_seen = get_triples(seen_graphs, distance=args.distance, train=False)
        print(len(train_triples_seen), len(test_triples_seen))
        test_triples_unseen = get_triples(unseen_graphs, distance=args.distance, train=False)
        if args.phase == TRAINING_PHASE:
            data = {
                'train': train_triples_seen,
                'test': test_triples_seen,
                'unseen': test_triples_unseen,
            }
        else:
            data = {
                'test': train_triples_seen + test_triples_seen,
            }

        for k, v in data.items():
            print(k, len(v))
        pretrained_lm_sequence_classification(data, args)
        
        ### Comment the break statement to train on all the folds
        break


if __name__ == '__main__':
    args = parse_args()
    main(args)
