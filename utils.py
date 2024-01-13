import json
import math
import os
import random
import tempfile
from streamlit.runtime.uploaded_file_manager import UploadedFile
import numpy as np
import re
from sklearn.metrics import roc_auc_score
import torch
from constants import *


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



def get_attr_name(uploaded_file):
    """
        This method returns the uploaded file from the streamlit file uploader
    """
    if isinstance(uploaded_file, UploadedFile):
        return uploaded_file.name
    return uploaded_file


def create_run_command_line(args):
    arguments = ['python', task2file_map[args.stage]]
    arguments += [f"--{k}={getattr(args, k)}" for k in vars(args) if getattr(args, k) is not None]
    return " ".join(arguments)


def create_run_config(args):
    """
        This method creates a run config for the given arguments
    """
    set_seed(args.seed)
    config = {k: get_attr_name(getattr(args, k)) for k in vars(args)}
    config[RUN_COMMAND] = create_run_command_line(args)

    print(config[RUN_COMMAND])
    
    file_name = f"{args.stage}_"
    if args.stage == PRETRAINING:
        if config[GPT_MODEL] in [UMLGPTMODEL]:
            file_name += f"{config[GPT_MODEL]}_tok={config['tokenizer']}"
        else:
            file_name += f"{config[GPT_MODEL]}"

    elif args.stage == UML_CLASSIFICATION:
        if args.classification_model not in [UMLGPTMODEL]:
            file_name += f"{os.path.basename(config[FROM_PRETRAINED])}"
        else:
            if FROM_PRETRAINED in config:
                file_name += f"{os.path.basename(config[FROM_PRETRAINED])}"
            else:
                file_name += f"{config[CLASSIFICATION_MODEL]}"

            file_name += f"_tok={config['tokenizer']}"
        
        file_name += f"_{config[CLASSIFICATION_TYPE]}"
        

    elif args.stage == LINK_PREDICTION:
        file_name += f"{os.path.basename(config[EMBEDDING_MODEL])}_tok={config['tokenizer']}"
    
    elif args.stage == ONTOML_CLS:
        file_name += f"_fp={config[FROM_PRETRAINED]}"
        file_name += f"_distance={args.distance}"
        file_name += f"_distance={args.exclude_limit}"

    
    os.makedirs(os.path.join(args.log_dir, file_name), exist_ok=True)
    args.log_dir = os.path.join(args.log_dir, file_name)
    
    os.makedirs(os.path.join(args.models_dir, file_name), exist_ok=True)
    args.models_dir = os.path.join(args.models_dir, file_name)

    args.config_file_name = file_name

    # print(config)
    # print(args.models_dir)

    json.dump(config, open(os.path.join(args.models_dir, f'config.json'), 'w'), indent=4)

    return config


def save_temporary_uploaded_file(uploaded_file, ext='.pt'):
    if uploaded_file is not None:
        # Create a temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmpfile:
            # Write the uploaded .pt file's content to the temporary file
            tmpfile.write(uploaded_file.getvalue())
            tmpfile_path = tmpfile.name
    
    return tmpfile_path



def get_plms(models_dir, task_type, model_name):
    plms = [
        f for f in os.listdir(models_dir) \
        if os.path.isdir(os.path.join(models_dir, f)) and \
        f.startswith(f'{task_type}_{model_name}')
    ]
    return plms