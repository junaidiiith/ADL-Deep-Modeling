import json
import math
import os
import random
import numpy as np
import re
from sklearn.metrics import roc_auc_score
import torch


clean_text = lambda x: re.sub(r'[^a-zA-Z0-9\s]', '', x).strip()

def get_recommendation_metrics(logits, labels):
    """
        This method calculates the following metrics for the given logits and labels
        1. MRR - Mean Reciprocal Rank
        2. Hits@1
        3. Hits@3
        4. Hits@5
        5. Hits@10
    """
    
    ## Check if logits and labels are numpy arrays
    if not isinstance(logits, np.ndarray):
        logits = logits.cpu().detach().numpy()
        labels = labels.cpu().detach().numpy()

    mrr = 0
    hits_at_1 = 0
    hits_at_3 = 0
    hits_at_5 = 0
    hits_at_10 = 0
    for i in range(len(logits)):
        logit = logits[i]
        label = labels[i]
        sorted_indices = np.argsort(logit)[::-1]
        rank = np.where(sorted_indices == label)[0][0] + 1
        mrr += 1/rank
        if rank == 1:
            hits_at_1 += 1
        if rank <= 3:
            hits_at_3 += 1
        if rank <= 5:
            hits_at_5 += 1
        if rank <= 10:
            hits_at_10 += 1
    
    mrr /= len(logits)
    hits_at_1 /= len(logits)
    hits_at_3 /= len(logits)
    hits_at_5 /= len(logits)
    hits_at_10 /= len(logits)
    
    return {
        'MRR': mrr,
        'Hits@1': hits_at_1,
        'Hits@3': hits_at_3,
        'Hits@5': hits_at_5,
        'Hits@10': hits_at_10,
    }


def get_recommendation_metrics_multi_label(logits, labels):
    """
        This method calculates the following metrics for the given logits and labels
        where labels is a n x k matrix of 0s and 1s where n is the number of samples and k is the number of classes
        logits allows for multiple classes to be predicted for each sample

        1. MRR - Mean Reciprocal Rank
        2. Hits@1
        3. Hits@3
        4. Hits@5
        5. Hits@10
    """
    if not isinstance(logits, np.ndarray):
        logits = logits.cpu().detach().numpy()
        labels = labels.cpu().detach().numpy()

    mrr = 0
    hits_at_1 = 0
    hits_at_3 = 0
    hits_at_5 = 0
    hits_at_10 = 0
    for i in range(len(logits)):
        logit = logits[i]
        allowed_labels = np.where(labels[i] == 1)[0]
        b_rank, b_hits_at_1, b_hits_at_3, b_hits_at_5, b_hits_at_10 = logit.shape[-1], 0, 0, 0, 0
        for label in allowed_labels:
            sorted_indices = np.argsort(logit)[::-1]
            rank = np.where(sorted_indices == label)[0][0] + 1
            b_rank = min(b_rank, rank)
            if rank == 1:
                b_hits_at_1 = 1
            if rank <= 3:
                b_hits_at_3 = 1
            if rank <= 5:
                b_hits_at_5 = 1
            if rank <= 10:
                b_hits_at_10 = 1

        mrr += 1/b_rank
        if rank == 1:
            hits_at_1 += b_hits_at_1
        if rank <= 3:
            hits_at_3 += b_hits_at_3
        if rank <= 5:
            hits_at_5 += b_hits_at_5
        if rank <= 10:
            hits_at_10 += b_hits_at_10
    
    mrr /= len(logits)
    hits_at_1 /= len(logits)
    hits_at_3 /= len(logits)
    hits_at_5 /= len(logits)
    hits_at_10 /= len(logits)
    
    return {
        'MRR': mrr,
        'Hits@1': hits_at_1,
        'Hits@3': hits_at_3,
        'Hits@5': hits_at_5,
        'Hits@10': hits_at_10,
    }


def compute_auc(pos_score, neg_score):
    """
        This method computes the AUC score for the given positive and negative scores
    """
    scores = torch.cat([pos_score, neg_score]).cpu().detach().numpy()
    labels = torch.cat(
        [torch.ones(pos_score.shape[0]), torch.zeros(neg_score.shape[0])]).cpu().detach().numpy()
    return roc_auc_score(labels, scores)


def get_eval_stats(eval_result):
    """
        This method computes the evaluation stats for the given eval result
    """
    stats = {
        'loss': eval_result['eval_loss'], 
        'perplexity': math.exp(eval_result['eval_loss']), 
        'accuracy': eval_result['eval_accuracy'],
    }
    return stats


def compute_metrics(eval_preds):
    """
        This method computes the metrics for the given eval preds
        This method is used as a callback in the Trainer class of the transformers library
    """
    logits, labels = eval_preds
    recommendation_metrics = get_recommendation_metrics(logits, labels)
    return recommendation_metrics


def set_seed(seed):
    """
        This method sets the seed for random, numpy, and torch libraries
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)


def get_package_name(obj):
    # For functions, classes, or methods
    if hasattr(obj, '__module__'):
        module_name = obj.__module__
        if hasattr(obj, '__class__') and obj.__class__.__name__ == 'function':
            # For functions
            return module_name
        else:
            # For classes or methods
            return f"{module_name}.{obj.__name__}"

    # For other objects (e.g., modules)
    return getattr(obj, '__name__', None)


def create_run_config(args):
    """
        This method creates a run config for the given arguments
    """
    set_seed(args.seed)
    config = {
        'batch_size': args.batch_size,
        'num_epochs': args.num_epochs,
        'lr': args.lr,
        'warmup_steps': args.warmup_steps,
        'embed_dim': args.embed_dim,
        'num_layers': args.num_layers,
        'num_heads': args.num_heads,
        'block_size': args.block_size,
        'class_type': args.class_type,
        'gpt_model': args.gpt_model,
        'tokenizer': args.tokenizer,
        'pooling': args.pooling,
        'classification_model': args.classification_model,
        'from_pretrained': args.from_pretrained,
    }
    
    file_name = f"{args.stage}_"
    if args.stage == 'pre':
        if args.gpt_model in ['uml-gpt']:
            file_name += f"{args.gpt_model}_tok={args.tokenizer}"
        else:
            file_name += f"{args.gpt_model}"

    elif args.stage == 'cls':
        if args.classification_model in ['uml-gpt']:
            file_name += f"{args.classification_model}_tok={args.tokenizer}"
        else:
            file_name += f"{args.classification_model}_tok={args.tokenizer}"
        file_name += f"_{args.class_type}"
        file_name += f"_{args.pooling}"
        file_name += f"_fp={args.from_pretrained if args.from_pretrained else '0'}"

    elif args.stage == 'lp':
        assert args.from_pretrained, "Pretrained model path is required for link prediction to get node embeddings"
        file_name += f"{args.gpt_model}_tok={args.tokenizer}"


    
    os.makedirs(os.path.join(args.log_dir, file_name), exist_ok=True)
    args.log_dir = os.path.join(args.log_dir, file_name)
    
    os.makedirs(os.path.join('results', file_name), exist_ok=True)
    args.results_dir = os.path.join('results', file_name)
    
    os.makedirs(os.path.join('models', file_name), exist_ok=True)
    args.models_dir = os.path.join('models', file_name)

    args.config_file_name = file_name

    json.dump(config, open(os.path.join(args.log_dir, f'config.json'), 'w'), indent=4)

    return config
