from torch.utils.data import Dataset
from tokenization import VocabTokenizer
from tokenization import get_encoding_size


class GenerativeUMLDataset(Dataset):
    """
    ``GenerativeUMLDataset`` class is a pytorch dataset for the generative task of UML
    """
    def __init__(self, data, tokenizer):
        """
        Args:
            data: list of triples (entity, relations, super_type)
            tokenizer: tokenizer to tokenize the data
        """
        
        super().__init__()
        self.data = data
        
        if isinstance(tokenizer, VocabTokenizer):
            self.inputs = tokenizer.batch_encode(data, return_tensors='pt', max_length='percentile')
        else:
            max_token_length = get_encoding_size(data, tokenizer)
            self.inputs = tokenizer(data, padding=True, return_tensors='pt', max_length=max_token_length, truncation=True)
        self.labels = self.inputs['input_ids'].clone()
        self.labels[self.labels == tokenizer.pad_token_id] = -100

        # print(self.labels[0].shape)

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return {
            'input_ids': self.inputs['input_ids'][idx],
            'attention_mask': self.inputs['attention_mask'][idx],
            'labels': self.labels[idx]
        }
    


class UMLNodeDataset(Dataset):
    """
    ``UMLNodeDataset`` class is a pytorch dataset for the classification task of UML
    """
    def __init__(self, data, tokenizer, label_map):
        super().__init__()
        self.data = data
        entity_inputs = [i[0] for i in data]
        entity_labels = [i[1] for i in data]
        if isinstance(tokenizer, VocabTokenizer):
            self.inputs = tokenizer.batch_encode(entity_inputs, return_tensors='pt', max_length='percentile')
        else:
            max_token_length = get_encoding_size(entity_inputs, tokenizer)
            self.inputs = tokenizer(entity_inputs, padding=True, return_tensors='pt', max_length=max_token_length, truncation=True)
        self.labels = [label_map[i] for i in entity_labels]
        self.i2c = {v: k for k, v in label_map.items()}
        self.num_classes = len(label_map)
        
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return {
            'input_ids': self.inputs['input_ids'][idx],
            'attention_mask': self.inputs['attention_mask'][idx],
            'labels': self.labels[idx]
        }

class EncodingsDataset(Dataset):
    """
    ``EncodingsDataset`` class is a pytorch dataset to create a dataset from the tokenized data
    """
    def __init__(self, tokenized):
        self.tokenized = tokenized
    
    def __len__(self):
        return len(self.tokenized['input_ids'])
    
    def __getitem__(self, index):
        item = {key: val[index] for key, val in self.tokenized.items()}
        return item
    
    @property
    def column_names(self):
        return self.tokenized.keys()
    
    

