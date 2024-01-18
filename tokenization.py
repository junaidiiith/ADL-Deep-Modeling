from transformers import AutoTokenizer
import os
import pickle
from constants import *
from tqdm.auto import tqdm
from common_utils import clean_text
import numpy as np
from nltk.tokenize import word_tokenize


def get_encoding_size(data, tokenizer):
    """
    ``get_encoding_size`` function returns the encoding size for the given data and tokenizer
        i.e., given a list of strings, it returns the 99.5th percentile of the lengths of the tokenized strings
    99.5th percentile is used to avoid the out of memory error while training the model
    """

    tokens = tokenizer(data)
    lengths = [len(i) for i in tokens['input_ids']]
    size = int(np.percentile(lengths, 99.5))
    # print("Encoding size: ", size)
    return size


def get_pretrained_lm_tokenizer(model_name, special_tokens=SPECIAL_TOKENS):
    """
        ``get_pretrained_lm_tokenizer`` function returns the tokenizer for the given hugging face language model
    """
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.add_special_tokens({'additional_special_tokens': special_tokens})
    if tokenizer.pad_token is None:
        tokenizer.pad_token = "<pad>"

    print("Vocab size: ", len(tokenizer))
    return tokenizer


def get_word_tokenizer_tokenizer(data, lower=True, special_tokens=SPECIAL_TOKENS):
    """
    ``get_word_tokenizer_tokenizer`` function constructs a custom Vocabulary tokenizer for the given data
    """

    tokenizer = VocabTokenizer(data, lower=lower, special_tokens=special_tokens)
    print("Vocab size: ", len(tokenizer))
    return tokenizer


def get_tokenization(tokenizer, data):
    """
    ``get_tokenization`` function returns the tokenization for the given tokenizer and data
    """
    if isinstance(tokenizer, VocabTokenizer):
        tokenized_data = tokenizer.batch_encode(
            data, return_tensors='pt', max_length='percentile')
    else:
        tokens = tokenizer(data)
        lengths = [len(i) for i in tokens['input_ids']]
        size = int(np.percentile(lengths, 99.5))

        tokenized_data = tokenizer(
            data, return_tensors='pt', padding=True, truncation=True, max_length=size)
    return tokenized_data


class VocabTokenizer:
    """
    class VocabTokenizer is a custom tokenizer that is used to tokenize the graph data
    It is used to tokenize the graph data for the generative task
    The vocabulary is constructed using the words in the graph data
    The encode, decode and batch_encode functions are used to encode, decode and batch encode the graph data
    The tokenizer takes special tokens as input and adds them to the vocabulary
    The special tokens are used to tokenize the graph data and treat them as special tokens
    """

    def __init__(self, data, lower=True, special_tokens=[]):
        self.lower = lower
        self.vocab = {}
        self.special_tokens = special_tokens
        
        for i in self.special_tokens:
            self.vocab[i] = len(self.vocab)

        for split_type in data:
            for i in tqdm(data[split_type]):
                word = " ".join(i) if isinstance(i, tuple) else i
                for j in word_tokenize(clean_text(word) if not self.lower else clean_text(word).lower()):
                    if j not in self.vocab:
                        self.vocab[j] = len(self.vocab)
        
        self.pad_token_id = self.vocab[PAD]
        self.pad_token = PAD

        self.unknown_token_id = self.vocab[UNK]
        self.unknown_token = UNK
        
        self.index_to_key = {v: k for k, v in self.vocab.items()}
    
    def batch_encode(self, x, return_tensors=None, max_length=None):
        """
        ``batch_encode`` function encodes the given list of strings

        Args:
            x: list of strings to be encoded
            return_tensors: whether to return tensors or lists
            max_length: maximum length of the encoded tokens
            max_length == 'percentile': maximum length is set to 99.95th percentile of the lengths of the encoded tokens
        """

        assert isinstance(x, list), "Input must be a list"
        # batch_encodings = [self.encode(i) for i in tqdm(x, desc='Encoding')]
        batch_encodings = [self.encode(i) for i in x]
        lengths = [len(i) for i in batch_encodings]
        perc_max_length = int(np.percentile(lengths, 99.95))
        max_length = 512 if max_length is None else (perc_max_length if max_length == 'percentile' else max_length)
        max_length = min(max_length, max([len(i) for i in batch_encodings]))
        
        batch_input_ids = [i[:min(max_length, len(i))] + [self.pad_token_id] * (max_length - min(max_length, len(i))) for i in batch_encodings]
        batch_attention_mask = [[1] * min(max_length, len(i)) + [0] * (max_length - min(max_length, len(i))) for i in batch_encodings]

        if return_tensors == 'pt':
            return {
                'input_ids': torch.LongTensor(batch_input_ids),
                'attention_mask': torch.LongTensor(batch_attention_mask)
            }
        elif return_tensors == 'np':
            return {
                'input_ids': np.array(batch_input_ids),
                'attention_mask': np.array(batch_attention_mask)
            }
        else:
            return {
                'input_ids': batch_input_ids,
                'attention_mask': batch_attention_mask
            }


    def encode(self, x, return_tensors=None):
        """
        ``encode`` function encodes the given string
        e.g.,
            input: "hello world"
            output: [1, 2]
        """
        input_ids = self(x)
        if return_tensors == 'pt':
            return torch.LongTensor(input_ids)
        elif return_tensors == 'np':
            return np.array(input_ids)
        return input_ids
    

    def __call__(self, x):
        if isinstance(x, tuple) or isinstance(x, list):
            x = " ".join(x)
        
        words, x = x.split(), list()
        for i in range(0, len(words)):
            if words[i] in self.special_tokens:
                x.append(words[i])
            else:
                x.extend(word_tokenize(clean_text(words[i]) if not self.lower else clean_text(words[i]).lower()))

        return [self.vocab.get(i, self.vocab['<unk>']) for i in x]
    
    def decode(self, x):
        """
        ``decode`` function decodes the given list of integers
        e.g.,
            input: [1, 2]
            output: "hello world"
        Inverse of encode function
        """
        assert isinstance(x, list), "Input must be a list"
        return [self.index_to_key[i] for i in x]
    
    def __len__(self):
        return len(self.vocab)
    
    def get_vocab(self):
        return self.vocab

    def add_special_tokens(self, special_tokens):
        for i in special_tokens:
            if i not in self.vocab:
                self.vocab[i] = len(self.vocab)
        self.index_to_key = {v: k for k, v in self.vocab.items()}
    
    def __str__(self) -> str:
        return f"VocabTokenizer(vsize={len(self.vocab)})"
    
    def save_pretrained(self, save_directory):
        """
        ``save_pretrained`` function saves the tokenizer to the given directory
        """
        if not os.path.exists(save_directory):
            os.makedirs(save_directory)
        tokenizer_file = os.path.join(save_directory, 'tokenizer.pkl')
        pickle.dump(self, open(tokenizer_file, 'wb'))
        print(f"Tokenizer saved to {tokenizer_file}")

    def __name__(self):
        return str(self)
    
    @property
    def name_or_path(self):
        return str(self)
