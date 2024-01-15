import numpy as np
import torch
import math
from sklearn.metrics import roc_auc_score
from constants import DEVICE


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
        # print(logits.shape, labels.shape, labels)
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


def compute_loss(pos_score, neg_score):
    scores = torch.cat([pos_score, neg_score]).to(DEVICE)
    labels = torch.cat([torch.ones(pos_score.shape[0]), torch.zeros(neg_score.shape[0])]).to(DEVICE)
    return torch.nn.BCEWithLogitsLoss()(scores.float(), labels.float())



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

