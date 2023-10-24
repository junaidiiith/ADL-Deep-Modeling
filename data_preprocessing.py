import torch
from transformers import AutoTokenizer
from tqdm.auto import tqdm
from itertools import chain


ROOT_TAG = '<MODEL>'
CLS_TAG = '<CLS>'
CLS_NAME_TAG = '<NAME>'
ATTRS_TAG = '<ATTRS>'
ASSOCS_TAG = '<ASSOCS>'
OPEN_CHAR = '('
CLOSE_CHAR = ')'
special_tags = [ROOT_TAG, CLS_TAG, CLS_NAME_TAG, ATTRS_TAG, ASSOCS_TAG, OPEN_CHAR, CLOSE_CHAR]


class GPT2Dataset(torch.utils.data.Dataset):
    def __init__(self, encoding):
        self.tokenized = encoding
        self.tokenizer = encoding
    
    def __len__(self):
        return len(self.tokenized['input_ids'])
    
    def __getitem__(self, index):
        item = {key: val[index] for key, val in self.tokenized.items()}
        return item


def get_model_from_tree_text(text):
    current_special = None
    i = 0
    classes = dict()
    l = text.split()
    while i < len(l):
        current_token = l[i]
        
        if current_token == CLS_NAME_TAG:
            current_special = CLS_TAG
            current_class = l[i+1]
            classes[current_class] = {'assocs': list(), 'attrs': list()}
            i += 1
        elif current_token == ASSOCS_TAG:
            current_special = ASSOCS_TAG
        elif current_token == ATTRS_TAG:
            current_special = ATTRS_TAG
        elif current_token not in special_tags:
            x, y = current_token, l[i+1]
            if current_special == ASSOCS_TAG:
                classes[current_class]['assocs'].append((x, y))
            else:
                classes[current_class]['attrs'].append((x, y))
            i += 1    
        i += 1
    return classes


def get_model_info_pmc(model_dict):
    classes = list(model_dict.keys())
    associations = list(chain.from_iterable([model_dict[c]['assocs'] for c in classes if len(model_dict[c]['assocs']) > 0]))
    attributes = list(chain.from_iterable([model_dict[c]['attrs'] for c in classes if len(model_dict[c]['attrs']) > 0]))

    associations = list(set(associations))
    attributes = list(set(attributes))

    dataset = {
        'classes': list(),
        'associations': list(),
        'attributes': list(),
    }
    all_attributes = " ".join([f"{x} {y}" for x, y in attributes])
    all_associations = " ".join([f"{x} {y}" for x, y in associations])
    all_classes = " ".join(classes)
    
    for i in range(len(classes)):
        input_classes = " ".join(classes[:i] + classes[i+1:])
        dataset['classes'].append((f"{input_classes} {all_associations} {all_attributes}", classes[i]))
    
    for i in range(len(associations)):
        input_associations = associations[:i] + associations[i+1:]
        input_associations = [f"{y}" for x, y in input_associations]
        input_associations = " ".join(input_associations)
        dataset['associations'].append((f"{all_classes} {input_associations} {all_attributes}", associations[i]))
    
    for i in range(len(attributes)):
        input_attributes = attributes[:i] + attributes[i+1:]
        input_attributes = [f"{y}" for x, y in input_attributes]
        input_attributes = " ".join(input_attributes)
        dataset['attributes'].append((f"{all_classes} {all_associations} {input_attributes}", attributes[i]))

    return dataset


def get_dataset_from_texts(texts):
    dataset = {
        'classes': list(),
        'associations': list(),
        'attributes': list(),
    }
    for text in tqdm(texts):
        model_dict = get_model_from_tree_text(text)
        model_dataset = get_model_info_pmc(model_dict)
        for key in model_dataset:
            dataset[key] += model_dataset[key]


def get_tokenizer(model_name):
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    if tokenizer.mask_token is None:
        tokenizer.add_special_tokens({'mask_token': '[MASK]'})
    
    if 'gpt' in model_name:
        tokenizer.padding_side = "left"
        tokenizer.pad_token = tokenizer.eos_token
    
    # tokenizer.add_special_tokens({'additional_special_tokens': [PATH_SEP]})
    
    return tokenizer


def get_tokenized(texts, max_tokens=512):
    tokenizer = get_tokenizer('gpt2')
    updated_texts = list()
    for text in tqdm(texts):
        i = 0
        txt_tokenized = tokenizer.encode(text)
        while i < len(txt_tokenized):
            tokenized_text = txt_tokenized[i:i+max_tokens]
            updated_texts.append(tokenizer.decode(tokenized_text))
            i += max_tokens
    
    tokenized_texts = tokenizer(updated_texts, padding=True, truncation=True, max_length=max_tokens, return_tensors='pt')
    print("Tokenization", tokenized_texts['input_ids'].shape)
    return tokenized_texts