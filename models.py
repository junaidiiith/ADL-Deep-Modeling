import torch
from transformers import AutoModel
import torch.nn as nn


device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


def weights_init(model):
    if isinstance(model, nn.Linear):
        nn.init.xavier_uniform_(model.weight.data)
        if model.bias is not None:
            nn.init.zeros_(model.bias.data)
    elif isinstance(model, nn.Embedding):
        nn.init.xavier_uniform_(model.weight.data)
    elif isinstance(model, nn.LayerNorm):
        nn.init.ones_(model.weight.data)
        nn.init.zeros_(model.bias.data)


class TripleClassifier(torch.nn.Module):
    """
        Classifier that first uses the model to get the pooled output
        Then applies a linear layer to get the logits for the entity classification
        Then applies another linear layer to get the logits for the super_type classification
    """

    def __init__(self, num_labels, num_spt_labels, mask_token, mask_token_id, model_name='xlm-roberta-base'):
        super(TripleClassifier, self).__init__()
        self.model = AutoModel.from_pretrained(model_name)
        self.num_labels = num_labels
        self.num_spt_labels = num_spt_labels
        self.dropout = torch.nn.Dropout(0.1)
        self.entity_linear = torch.nn.Linear(self.model.config.hidden_size, self.num_labels)
        self.spt_linear = torch.nn.Linear(self.model.config.hidden_size, self.num_spt_labels)
        self.softmax = torch.nn.Softmax(dim=1)
        self.mask_token_id = mask_token_id
        self.mask_token = mask_token
        
    
    def forward(self, batch):
        entity_logits = self.get_entity_logits(batch['entity_input_ids'], batch['entity_attention_mask'])
        spt_logits = self.get_super_type_logits(batch['super_type_input_ids'], batch['super_type_attention_mask'])
        return entity_logits, spt_logits


    def get_entity_logits(self, input_ids, attention_mask):

        """
            This method returns the logits by taking the logits of the [MASK] token and applying a linear layer
            for the entity classification task
        """
            
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
        masked_token_embedding = outputs[0][:, 1, :]
        masked_token_embedding = self.dropout(masked_token_embedding)
        entity_logits = self.entity_linear(masked_token_embedding)
        return entity_logits


    def get_super_type_logits(self, input_ids, attention_mask):
        """
            This method returns the logits by applying a linear layer on the CLS token embedding
            for the super_type classification task
        """

        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
        cls_token_embedding = outputs[0][:, 0, :]
        cls_token_embedding = self.dropout(cls_token_embedding)
        spt_logits = self.spt_linear(cls_token_embedding)
        return self.softmax(spt_logits)


    def get_entity_loss(self, logits, labels):
        """
            logits: (batch_size, num_labels)
            labels: (batch_size)
            This method calculates the loss for the entity classification using cross entropy loss
        """
        logits = logits.to(device)
        labels = labels.to(device)

        loss_fct = torch.nn.CrossEntropyLoss()
        loss = loss_fct(logits, labels)
        return loss

    def get_super_type_loss(self, spt_logits, spt_labels):
        """
            spt_logits: (batch_size, num_spt_labels)
            spt_labels: (batch_size, num_spt_labels)
            This method calculates the loss for the super_type classification such that,
            if spt_labels shape is (batch_size, num_spt_labels), then the loss is calculated using cross entropy loss
            else if spt_labels shape is (batch_size, num_spt_labels, k), then the loss is calculated using binary cross entropy loss
        """

        spt_logits = spt_logits.to(device)
        spt_labels = spt_labels.to(device)

        if len(spt_labels.shape) == 1:
            loss_fct = torch.nn.CrossEntropyLoss()
            loss = loss_fct(spt_logits, spt_labels)
        else:
            loss_fct = torch.nn.BCELoss()
            loss = loss_fct(spt_logits.float(), spt_labels.float())
        return loss
    

    def get_loss(self, logits, spt_logits, labels, spt_labels):
        entity_loss = self.get_entity_loss(logits, labels)
        spt_loss = self.get_super_type_loss(spt_logits, spt_labels)
        return entity_loss, spt_loss
    


class Head(nn.Module):
    """ one head of self-attention """

    def __init__(self, embed_dim, head_size, dropout=0.1):
        super().__init__()
        self.key = nn.Linear(embed_dim, head_size, bias=False)
        self.query = nn.Linear(embed_dim, head_size, bias=False)
        self.value = nn.Linear(embed_dim, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(head_size, head_size)))
        self.softmax = nn.Softmax(dim=-1)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x, attention_mask):
        B, T, C = x.shape
        k = self.key(x)   # (B, T, C)
        q = self.query(x) # (B, T, C)

        # Compute attention scores ("affinities") only where the mask is non-zero
        wei = q @ k.transpose(-2, -1) * C**-0.5  # (B, T, C) @ (B, C, T) -> (B, T, T)
        wei = wei.masked_fill((attention_mask.unsqueeze(1) == 0), float('-inf'))  # (B, T, T)
        wei = self.softmax(wei)  # (B, T, T)
        wei = self.dropout(wei)

        # Perform the weighted aggregation of the values
        v = self.value(x)  # (B, T, C)
        out = wei @ v  # (B, T, T) @ (B, T, C) -> (B, T, C)
        return out


class MultiHeadAttention(nn.Module):
    """ multiple heads of self-attention in parallel """

    def __init__(self, embed_dim, num_heads, dropout=0.1):
        super().__init__()
        head_size = embed_dim // num_heads
        self.heads = nn.ModuleList([Head(embed_dim, head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, attn_mask):
        out = torch.cat([h(x, attn_mask) for h in self.heads], dim=-1)
        out = self.dropout(self.proj(out))
        return out

class FeedFoward(nn.Module):
    """ a simple linear layer followed by a non-linearity """

    def __init__(self, embed_dim, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(embed_dim, 4 * embed_dim),
            nn.ReLU(),
            nn.Linear(4 * embed_dim, embed_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)

class Block(nn.Module):
    """ Transformer block: communication followed by computation """

    def __init__(self, embed_dim, n_head):
        # embed_dim: embedding dimension, n_head: the number of heads we'd like
        super().__init__()
        self.sa = MultiHeadAttention(embed_dim, n_head)
        self.ffwd = FeedFoward(embed_dim)
        self.ln1 = nn.LayerNorm(embed_dim)
        self.ln2 = nn.LayerNorm(embed_dim)

    def forward(self, x, attn_mask):
        x = x + self.sa(self.ln1(x), attn_mask)
        x = x + self.ffwd(self.ln2(x))
        return x


class UMLGPT(nn.Module):

    def __init__(self, vocab_size, embed_dim, block_size, n_layer, n_head):
        super().__init__()
        # each token directly reads off the logits for the next token from a lookup table
        self.token_embedding_table = nn.Embedding(vocab_size, embed_dim)
        self.position_embedding_table = nn.Embedding(block_size, embed_dim)
        self.blocks = nn.Sequential(*[Block(embed_dim, n_head) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(embed_dim) # final layer norm
        self.lm_head = nn.Linear(embed_dim, vocab_size)

        self.apply(weights_init)

    def forward(self, x, attention_mask):
        # x: [batch_size, seq_len]
        # attention_mask: [batch_size, seq_len]
        token_embeddings = self.token_embedding_table(x)
        position_ids = torch.arange(x.size(1), dtype=torch.long, device=x.device)
        position_ids = position_ids.unsqueeze(0).expand_as(x)
        position_embeddings = self.position_embedding_table(position_ids)
        embeddings = token_embeddings + position_embeddings

        # # Modify the forward pass to include src_key_padding_mask
        for block in self.blocks:
            embeddings = block(embeddings, attention_mask)

        embeddings = self.ln_f(embeddings)  
        logits = self.lm_head(embeddings)
        return logits
