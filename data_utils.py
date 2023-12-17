from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import MultiLabelBinarizer
from torch.utils.data import Dataset
import evaluate
from sklearn.metrics import accuracy_score
import torch
import re
from tqdm.auto import tqdm
from collections import defaultdict
import numpy as np



vector_file_prefix = "word2vec_models/vectors_"


SEP = "[SEP]"

e_s = {'rel': 'relates', 'gen': 'generalizes'}
remove_extra_spaces = lambda txt: re.sub(r'\s+', ' ', txt.strip())

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



class TripleDataset(Dataset):
    def __init__(self, triples, entity_map, super_type_map, tokenizer, multi_label=True):
        self.labels = torch.from_numpy(np.array([entity_map[t[0]] for t in triples]))
        self.super_type_labels = get_super_type_labels(triples, super_type_map, multi_label)
        
        entity_triples = [f'{tokenizer.mask_token} {t[1]} | {t[2]}'.strip() for t in triples]
        max_token_length, _, _ = get_encoding_size(entity_triples, tokenizer)
        self.entity_tokenized = tokenizer(
            entity_triples, padding=True, return_tensors='pt', max_length=max_token_length, truncation=True)


        super_type_triples = [f'{t[0]} {t[1]}'.strip() for t in triples]
        max_token_length, _, _ = get_encoding_size(super_type_triples, tokenizer)
        self.super_type_tokenized = tokenizer(
            super_type_triples, padding=True, return_tensors='pt', max_length=max_token_length, truncation=True)
        
        self.entity_mask = torch.from_numpy(np.array([len(t[1].strip()) != 0 for t in triples]))
        self.super_type_mask = torch.from_numpy(np.array([len(t[2].strip()) != 0 for t in triples]))

        self.entity_map = entity_map
        self.super_type_map = super_type_map

    @property
    def num_labels(self):
        return len(self.entity_map)

    @property
    def num_super_type_labels(self):
        return len(self.super_type_map)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        inputs = {
            'entity_input_ids': self.entity_tokenized['input_ids'][idx],
            'entity_attention_mask': self.entity_tokenized['attention_mask'][idx],
            'super_type_input_ids': self.super_type_tokenized['input_ids'][idx],
            'super_type_attention_mask': self.super_type_tokenized['attention_mask'][idx],
            'entity_mask': self.entity_mask[idx],
            'super_type_mask': self.super_type_mask[idx],
            'entity_label': self.labels[idx],
            'super_type_label': self.super_type_labels[idx],
        }

        return inputs


class TaskTypeDataset(Dataset):
    def __init__(self, inputs, task_type='entity'):
        mask = inputs[f'{task_type}_mask']
        self.labels = inputs[f'{task_type}_label'][mask]
        self.input_ids = inputs[f'{task_type}_input_ids'][mask]
        self.attention_mask = inputs[f'{task_type}_attention_mask'][mask]

    
    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        inputs = {
            'input_ids': self.input_ids[idx],
            'attention_mask': self.attention_mask[idx],
            'labels': self.labels[idx],
        }

        return inputs




accuracy = evaluate.load("accuracy")
def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    return accuracy.compute(predictions=predictions, references=labels)


def init_glove_model(model_type):
    f_vec = f"{vector_file_prefix}{model_type}.txt"
    vocab, W = init_glove(f_vec)
    return vocab, W


def get_encoding_size(triples, tokenizer, allowed_perc=0.005):
    lengths = [len(tokenizer.encode(t)) for t in triples]
    return int(np.percentile(lengths, 100 - allowed_perc)), np.max(lengths), np.mean(lengths)


def get_accuracy(tokenizer, mask_filler_pipe, dataset):
    decoded_texts = list()
    for data in dataset:
        decoded_texts.append((data['labels'], tokenizer.decode(data['input_ids'])))
    count = 0
    all_preds, all_actuals = list(), list()
    for txt_labels, txt_batch in tqdm(decoded_texts):
        try:
            fill_out = mask_filler_pipe(txt_batch)
            preds = [x[0]['token'] for x in fill_out]
            actuals = txt_labels[torch.where(txt_labels != -100)].tolist()
            all_preds += preds
            all_actuals += actuals
            count += preds == actuals
        except:
            continue
  
    accuracy = count / len(decoded_texts)
    masked_accuracy = accuracy_score(all_actuals, all_preds)
    return f"{accuracy:.4f}", f"{masked_accuracy:.4f}"


def update_encodings(encodings, current_token, update_token, pad_token="[PAD]"):
    updated_encodings = {k: list() for k in ['input_ids', 'attention_mask', 'labels']}
    for ips in tqdm(encodings['input_ids'], desc='Adding [MASK] tokens'):
        input_ids, attention_mask, labels = get_updated_encoding(ips, current_token, update_token)
        updated_encodings['input_ids'].append(torch.tensor(input_ids))
        updated_encodings['attention_mask'].append(torch.tensor(attention_mask))
        updated_encodings['labels'].append(torch.tensor(labels))

    updated_encodings['input_ids'] = pad_1d_tensors(updated_encodings['input_ids'], pad_token)
    updated_encodings['attention_mask'] = pad_1d_tensors(updated_encodings['attention_mask'], 0)
    updated_encodings['labels'] = pad_1d_tensors(updated_encodings['labels'], -100)
    return updated_encodings


def get_updated_encoding(input_ids, current_token, update_token):
    i = 0
    updated_input_ids, labels = list(), list()
    while i < len(input_ids):
        curr_token = input_ids[i]
        if curr_token == current_token:
            i += 1
            while i < len(input_ids) and input_ids[i] != current_token:
                labels.append(input_ids[i])
                updated_input_ids.append(update_token)
                i += 1
        else:
            updated_input_ids.append(curr_token)
            labels.append(-100)
        i += 1
    attention_mask = [1] * len(updated_input_ids)
    return updated_input_ids, attention_mask, labels


def init_glove(vectors_file, normalize=False):
    with open(vectors_file, 'r') as f:
        vectors = {}
        for line in f:
            vals = line.rstrip().split(' ')
            vectors[vals[0]] = [float(x) for x in vals[1:]]

    words = list(vectors.keys())
    vocab_size = len(words)
    vocab = {w: idx for idx, w in enumerate(words)}
    inv_vocab = {idx: w for idx, w in enumerate(words)}
    vector_dim = len(vectors[inv_vocab[0]])
    W_ = np.zeros((vocab_size, vector_dim))

    for word, v_ in vectors.items():
        if word == '<unk>':
            continue
        W_[vocab[word], :] = v_

    if normalize:
        d = (np.sum(W_ ** 2, 1) ** 0.5)
        W_ = (W_.T / d).T
    return vocab, W_


def get_w2v_embeddings(graph, w2v_model, uncased=True):
    vocab, W = w2v_model
    # embed_dim = W.shape[1]
    # node_embeddings = torch.randn(len(graph.nodes), embed_dim)
    # return node_embeddings
    node_embeddings = list()
    for n in graph.nodes:
        name = graph.nodes[n]['name'] if graph.nodes[n]['name'] != "Null" else "relates"
        name = name.lower()
        node_type = graph.nodes[n]['type'] if not uncased else graph.nodes[n]['type'].lower()
        name += f" {node_type}" 
        name_parts = name.split()
        embeddings = list()
        for part in name_parts:
            if part in vocab:
                embeddings.append(W[vocab[part]])
            else:
                embeddings.append(np.zeros(W.shape[1]))
        if len(embeddings) == 0:
            embeddings.append(np.zeros(W.shape[1]))
    
        embeddings = np.array(embeddings)
        embeddings = np.mean(embeddings, axis=0)

        node_embeddings.append(embeddings)
    node_embeddings = np.array(node_embeddings)
    return torch.from_numpy(node_embeddings)


def pad_1d_tensors(tensors_list, padding_token):
    max_length = max(tensor.shape[0] for tensor in tensors_list)
    padded_tensors = []

    # Pad each tensor to match the maximum size along the specified dimension
    for tensor in tensors_list:
        sizes_to_pad = [max_length - s for s in tensor.shape]
        padding = torch.full(sizes_to_pad, padding_token, dtype=tensor.dtype)
        padded_tensor = torch.cat([tensor, padding], dim=0)
        padded_tensors.append(padded_tensor)

    return torch.stack(padded_tensors)

# if __name__ == '__main__':
#     with open('datasets/ecore_graph_pickles/combined_graphs.pkl', 'rb') as f:
#         graphs = pickle.load(f)

