import json
import os
import torch
import torch_geometric
import torch.nn as nn


from constants import DEVICE


def weights_init(model):
    """
    Initialize the weights of the model
    xaiver_uniform is used for linear layers and embeddings
    zeros is used for biases
    xavier_uniform initializes the weights with a uniform distribution
    This is done to avoid the exploding gradient problem
    """

    if isinstance(model, nn.Linear):
        nn.init.xavier_uniform_(model.weight.data)
        if model.bias is not None:
            nn.init.zeros_(model.bias.data)
    elif isinstance(model, nn.Embedding):
        nn.init.xavier_uniform_(model.weight.data)
    elif isinstance(model, nn.LayerNorm):
        nn.init.ones_(model.weight.data)
        nn.init.zeros_(model.bias.data)


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
        """
        x: [batch_size, seq_len, embed_dim]
        attention_mask: [batch_size, seq_len]

        This method computes the attention scores between each token in the sequence
        """
        _, _, C = x.shape
        k = self.key(x)
        q = self.query(x)

        # Compute attention scores ("affinities") only where the mask is non-zero
        wei = q @ k.transpose(-2, -1) * C**-0.5
        wei = wei.masked_fill((attention_mask.unsqueeze(1) == 0), float('-inf'))
        wei = self.softmax(wei)
        wei = self.dropout(wei)

        # Perform the weighted aggregation of the values
        v = self.value(x)
        out = wei @ v
        return out


class MultiHeadAttention(nn.Module):
    """
    multiple heads of self-attention in parallel
    This class first splits the embedding dimension into multiple heads
    Then, each head computes the attention scores between each token in the sequence
    Finally, the outputs of all the heads are concatenated and projected back to the original embedding dimension
    """

    def __init__(self, embed_dim, num_heads, dropout=0.1):
        super().__init__()
        head_size = embed_dim // num_heads
        self.heads = nn.ModuleList([Head(embed_dim, head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, attn_mask):
        """
        x: [batch_size, seq_len, embed_dim]
        """
        out = torch.cat([h(x, attn_mask) for h in self.heads], dim=-1)
        out = self.dropout(self.proj(out))
        return out

class FeedFoward(nn.Module):
    """
    a simple linear layer followed by a non-linearity
    """

    def __init__(self, input_dim, embed_dim=None, num_classes=None, dropout=0.1):
        super().__init__()

        if num_classes is None:
            num_classes = input_dim if embed_dim is None else embed_dim

        if embed_dim is None:
            embed_dim = input_dim
        
        
        self.net = nn.Sequential(
            nn.Linear(input_dim, 4 * embed_dim),
            nn.ReLU(),
            nn.Linear(4 * embed_dim, num_classes),
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
    """
    UML-GPT model

    vocab_size: the size of the vocabulary
    embed_dim: the embedding dimension
    block_size: the maximum sequence length
    n_layer: the number of transformer blocks
    n_head: the number of heads in each transformer block
    load_pretrained_from: the path to the pretrained model

    This class uses the string representation of the node as the input
    The string representation is tokenized using the tokenizer
    The tokenized sequence is then passed through the transformer blocks
    Finally, the logits for the next token are computed using a linear layer
    
    """
    def __init__(self, vocab_size, embed_dim, block_size, n_layer, n_head, load_pretrained_from=None):
        super().__init__()
        # each token directly reads off the logits for the next token from a lookup table

        if load_pretrained_from is not None:
            self.load_pretrained(load_pretrained_from)
        else:
            self.token_embedding_table = nn.Embedding(vocab_size, embed_dim)
            self.position_embedding_table = nn.Embedding(block_size, embed_dim)
            self.blocks = nn.Sequential(*[Block(embed_dim, n_head) for _ in range(n_layer)])
            self.ln_f = nn.LayerNorm(embed_dim) # final layer norm
            self.lm_head = nn.Linear(embed_dim, vocab_size)

            self.apply(weights_init)


    def forward(self, x, attention_mask):
        """
        x: [batch_size, seq_len]
        attention_mask: [batch_size, seq_len]

        This method computes the logits for the next token
        """
        embeddings = self.get_embedding(x, attention_mask)
        logits = self.lm_head(embeddings)
        return logits


    def get_loss(self, logits, labels, ignore_index=-100):
        """
        logits: [batch_size, seq_len, vocab_size]
        labels: [batch_size, seq_len]

        This method computes the loss for the next token prediction task
        This is achieved by shifting the labels by one position and computing the cross entropy loss
        """
        block_size = self.position_embedding_table.weight.shape[0]
        labels = labels[..., :block_size]
        loss = None
        if labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = nn.CrossEntropyLoss(ignore_index=ignore_index)
            loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
        
        return loss
    
    
    def get_embedding(self, x, attention_mask):
        """
        x: [batch_size, seq_len]
        attention_mask: [batch_size, seq_len]
        """
        block_size = self.position_embedding_table.weight.shape[0]
        vocab_size = self.token_embedding_table.weight.shape[0]

        x = x[..., :block_size]
        attention_mask = attention_mask[..., :block_size]

        try:
            assert x.shape[-1] <= block_size
        except AssertionError as e:
            print("Block size is too small for the text!")
            raise e

        # print("Token embeddings", x.shape, torch.min(x), torch.max(x), vocab_size)
        try:
            assert torch.min(x) <= vocab_size
            assert torch.max(x) <= vocab_size
            
        except AssertionError as e:
            print("Fix the token embedding shape issue bro!")
            raise e
        
        token_embeddings = self.token_embedding_table(x)

        position_ids = torch.arange(x.size(1), dtype=torch.long, device=x.device)
        position_ids = position_ids.unsqueeze(0).expand_as(x)

        # print("Position embeddings", position_ids.shape, torch.min(position_ids), torch.max(position_ids), block_size)
        try:
            torch.min(position_ids) <= block_size
            torch.max(position_ids) <= block_size
        except AssertionError as e:
            print("Fix the position embeddings shape bro!")
            raise e
        position_embeddings = self.position_embedding_table(position_ids)
        

        embeddings = token_embeddings + position_embeddings

        # # Modify the forward pass to include src_key_padding_mask


        for block in self.blocks:
            # print("Embed dim: ", embeddings.shape)
            embeddings = block(embeddings, attention_mask)

        embeddings = self.ln_f(embeddings)
        return embeddings


    def get_model_size(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def __repr__(self):
        return super().__repr__() + f'\nNumber of parameters: {self.get_model_size() / 1000000:.3f}M'
    
    
    @property
    def __name__(self):
        return 'UMLGPT'
    

    @property
    def name_or_path(self):
        return 'UMLGPT'
    

    @staticmethod
    def from_pretrained(state_dict_pth):
        state_dict = torch.load(state_dict_pth, map_location=DEVICE)
        vocab_size, embed_dim = [s.shape for _, s in state_dict.items() if 'token_embedding_table' in _][0]
        num_heads = max([int(name.split('.sa.heads.')[1].split('.')[0]) for name, s in state_dict.items() if '.sa.heads.' in name]) + 1
        block_size = [s.shape[0] for _, s in state_dict.items() if 'position_embedding_table' in _][0]
        num_layers = max([int(name.split('blocks.')[1].split('.')[0]) for name, s in state_dict.items() if 'blocks.' in name]) + 1
        model = UMLGPT(vocab_size, embed_dim, block_size, num_layers, num_heads)
        model.load_state_dict(state_dict)
        return model
    
    
class UMLGPTClassifier(nn.Module):
    """
    UML-GPT model for classification

    model: the UML-GPT model
    num_classes: the number of classes

    """
    def __init__(self, model, num_classes):
        super().__init__()
        
        self.model = model
        _, embed_dim = self.model.lm_head.weight.data.shape
        self.classifier = FeedFoward(input_dim=embed_dim, num_classes=num_classes)
        self.apply(weights_init)

    def forward(self, x, attention_mask, pool=None):
        # x: [batch_size, seq_len]
        # attention_mask: [batch_size, seq_len]
        lm_logits = self.model.get_embedding(x, attention_mask)
        if pool:
            """Pool the logits across the sequence dimension"""
            lm_logits = torch.mean(lm_logits, dim=1)
        else:
            """Use the logits at the last position"""
            lm_logits = lm_logits[:, -1, :]
        
        logits = self.classifier(lm_logits)
        return logits
    
    def get_loss(self, logits, labels):
        logits = logits.to(DEVICE)
        labels = labels.to(DEVICE)

        if len(labels.shape) == 1:
            loss_fct = torch.nn.CrossEntropyLoss()
            loss = loss_fct(logits, labels)
        else:
            loss_fct = torch.nn.BCEWithLogitsLoss()
            loss = loss_fct(logits.float(), labels.float())
        return loss
    
    def get_embedding(self, x, attention_mask):
        return self.model.get_embedding(x, attention_mask)

    def get_model_size(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def __repr__(self):
        return super().__repr__() + f'\nNumber of parameters: {self.get_model_size()/1000000:.3f}M'
    
    @staticmethod
    def from_pretrained(state_dict_path, num_classes, init_classifier=True):
        if init_classifier:
            print("Initializing classifier from pretrained model with num classes: ", num_classes)
            model = UMLGPTClassifier(UMLGPT.from_pretrained(state_dict), num_classes)
        else:
            state_dict = torch.load(state_dict_path, map_location=DEVICE)
            vocab_size, embed_dim = [s.shape for _, s in state_dict.items() if 'token_embedding_table' in _][0]
            num_heads = max([int(name.split('.sa.heads.')[1].split('.')[0]) for name, s in state_dict.items() if '.sa.heads.' in name]) + 1
            block_size = [s.shape[0] for _, s in state_dict.items() if 'position_embedding_table' in _][0]
            num_layers = max([int(name.split('blocks.')[1].split('.')[0]) for name, s in state_dict.items() if 'blocks.' in name]) + 1
            num_classes = state_dict['classifier.net.2.weight'].shape[0]
            uml_gpt = UMLGPT(vocab_size, embed_dim, block_size, num_layers, num_heads)

            model = UMLGPTClassifier(uml_gpt, num_classes)
            model.load_state_dict(state_dict)
            
        return model


class GNNModel(torch.nn.Module):
    """
        A general GNN model created using the PyTorch Geometric library
        model_name: the name of the GNN model
        input_dim: the input dimension
        hidden_dim: the hidden dimension
        out_dim: the output dimension

        num_layers: the number of GNN layers
        num_heads: the number of heads in the GNN layer
        residual: whether to use residual connections
        l_norm: whether to use layer normalization
        dropout: the dropout probability
    
    """
    def __init__(self, model_name, input_dim, hidden_dim, out_dim, num_layers, num_heads=None, residual=False, l_norm=False, dropout=0.1):
        super(GNNModel, self).__init__()

        self.input_dim = input_dim
        self.embed_dim = hidden_dim
        self.out_dim = out_dim
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.residual = residual
        self.l_norm = l_norm
        self.dropout = dropout
        

        gnn_model = getattr(torch_geometric.nn, model_name)
        self.conv_layers = nn.ModuleList()
        if model_name == 'GINConv':
            input_layer = gnn_model(nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.ReLU()), train_eps=True)
        elif num_heads is None:
            input_layer = gnn_model(input_dim, hidden_dim, aggr='SumAggregation')
        else:
            input_layer = gnn_model(input_dim, hidden_dim, heads=num_heads, aggr='SumAggregation')
        self.conv_layers.append(input_layer)

        for _ in range(num_layers - 2):
            if model_name == 'GINConv':
                self.conv_layers.append(gnn_model(nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.ReLU()), train_eps=True))
            elif num_heads is None:
                self.conv_layers.append(gnn_model(hidden_dim, hidden_dim, aggr='SumAggregation'))
            else:
                self.conv_layers.append(gnn_model(num_heads*hidden_dim, hidden_dim, heads=num_heads, aggr='SumAggregation'))

        if model_name == 'GINConv':
            self.conv_layers.append(gnn_model(nn.Sequential(nn.Linear(hidden_dim, out_dim), nn.ReLU()), train_eps=True))
        else:
            self.conv_layers.append(gnn_model(hidden_dim if num_heads is None else num_heads*hidden_dim, out_dim, aggr='SumAggregation'))
            
        self.activation = nn.ReLU()
        self.layer_norm = nn.LayerNorm(hidden_dim if num_heads is None else num_heads*hidden_dim) if l_norm else None
        self.residual = residual
        self.dropout = nn.Dropout(dropout)


    def forward(self, in_feat, edge_index):
        h = in_feat
        h = self.conv_layers[0](h, edge_index)
        h = self.activation(h)
        if self.layer_norm is not None:
            h = self.layer_norm(h)
        h = self.dropout(h)

        for conv in self.conv_layers[1:-1]:
            h = conv(h, edge_index) if not self.residual else conv(h, edge_index) + h
            h = self.activation(h)
            if self.layer_norm is not None:
                h = self.layer_norm(h)
            h = self.dropout(h)
        
        h = self.conv_layers[-1](h, edge_index)
        return h
  
    def save_pretrained(self, path):
        os.makedirs(path, exist_ok=True)
        state_dict_path = f'{path}/gnn_state_dict.pt'
        config_path = f'{path}/gnn_config.json'
        with open(config_path, 'w') as f:
            json.dump(
                {
                    "model_name":"SAGEConv", 
                    "input_dim": self.input_dim,
                    "embed_dim": self.embed_dim,
                    "out_dim": self.out_dim,
                    "num_layers": self.num_layers,
                    "num_heads": self.num_heads,
                    "residual": self.residual,
                    "l_norm": self.l_norm,
                    "dropout": self.dropout.p,
                }, f)
    
        torch.save(self.state_dict(), state_dict_path)
        print(f'Saved GNN model at {path}')


    @staticmethod
    def from_pretrained(state_dict_dir):
        state_dict = torch.load(f"{state_dict_dir}/gnn_state_dict.pt", map_location=DEVICE)
        gnn_model_config = json.load(open(f"{state_dict_dir}/gnn_config.json", 'r'))
        gnn_model = GNNModel(
            model_name=gnn_model_config['model_name'],
            input_dim=gnn_model_config['input_dim'],
            hidden_dim=gnn_model_config['embed_dim'],
            out_dim=gnn_model_config['out_dim'],
            num_heads=gnn_model_config['num_heads'],
            num_layers=gnn_model_config['num_layers'],
            residual=gnn_model_config['residual'],
            dropout=gnn_model_config['dropout'],
        )
        gnn_model.load_state_dict(state_dict)
        return gnn_model
  

class MLPPredictor(nn.Module):

    """
    An MLP predictor for link prediction

    h_feats: the input dimension
    num_classes: the number of classes
    num_layers: the number of layers in the MLP

    This class concatenates the node embeddings of the two nodes in the edge
    The concatenated embeddings are then passed through an MLP
    """

    def __init__(self, h_feats, num_layers=2, num_classes=1):
        super().__init__()
        self.layers = nn.ModuleList()
        self.embed_dim = h_feats
        self.num_layers = num_layers
        self.num_classes = num_classes

        in_feats = h_feats * 2
        for _ in range(num_layers - 1):
            self.layers.append(nn.Linear(in_feats, h_feats))
            self.layers.append(nn.ReLU())
            in_feats = h_feats
        
        self.layers.append(nn.Linear(h_feats, num_classes))


    def forward(self, x, edge_index):
        h = torch.cat([x[edge_index[0]], x[edge_index[1]]], dim=-1)
        for layer in self.layers:
            h = layer(h)
        
        h = h.squeeze(1)
        return h

    def save_pretrained(self, pth):
        predictor_state_dict_path = f'{pth}/predictor_state_dict.pt'
        torch.save(self.state_dict(), predictor_state_dict_path)

        config_path = f'{pth}/predictor_config.json'
        with open(config_path, 'w') as f:
            json.dump(
                {
                    "h_feats": self.embed_dim,
                    "num_classes": self.num_classes,
                    "num_layers": self.num_layers,
                }, f)
        
        print(f'Saved MLP predictor at {pth}')
    
    @staticmethod
    def from_pretrained(state_dict_dir):
        state_dict = torch.load(f"{state_dict_dir}/predictor_state_dict.pt", map_location=DEVICE)
        predictor_config = json.load(open(f"{state_dict_dir}/predictor_config.json", 'r'))
        predictor = MLPPredictor(
            h_feats=predictor_config['h_feats'],
            num_classes=predictor_config['num_classes'],
            num_layers=predictor_config['num_layers'],
        )
        predictor.load_state_dict(state_dict)
        return predictor