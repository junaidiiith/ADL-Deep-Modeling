import torch
from uml_datasets import EncodingsDataset
from models import UMLGPT, UMLGPTClassifier


from constants import DEVICE



def get_embedding(model, encodings, pooling='last'):
    """
    ``get_embedding`` function returns the embeddings for the given model and encodings
    pooling: last, mean, max, min, sum, cls
    pooling is used to pool the embeddings of the tokens in the sequence
    e.g., if pooling is last, the last token embedding is used as the embedding for the sequence
    if pooling is mean, the mean of the token embeddings is used as the embedding for the sequence
    """
    encoding_dataset = EncodingsDataset(encodings)
    encoding_dataloader = torch.utils.data.DataLoader(encoding_dataset, batch_size=128, shuffle=False)
    model.eval()

    with torch.no_grad():
        embeddings = list()
        for batch in encoding_dataloader:

            if isinstance(model, UMLGPT) or isinstance(model, UMLGPTClassifier):
                outputs = model.get_embedding(batch['input_ids'].to(DEVICE), batch['attention_mask'].to(DEVICE))
            else:
                encodings = {k: v.to(DEVICE) for k, v in batch.items()}
                outputs = model(**encodings)[0]

            outputs = outputs.cpu().detach()
            if pooling == 'last':
                outputs = outputs[:, -1, :]
            elif pooling == 'mean':
                outputs = torch.mean(outputs, dim=1)
            elif pooling == 'max':
                outputs = torch.max(outputs, dim=1)[0]
            elif pooling == 'min':
                outputs = torch.min(outputs, dim=1)[0]
            elif pooling == 'sum':
                outputs = torch.sum(outputs, dim=1)
            elif pooling == 'cls':
                outputs = outputs[:, 0, :]
            else:
                raise ValueError(f"Pooling {pooling} not supported")
            embeddings.append(outputs)
        
        embeddings = torch.cat(embeddings, dim=0)
        
    return embeddings