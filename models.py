import torch
from transformers import AutoModel


device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

class TripleClassifier(torch.nn.Module):
    """
        Classifier that first uses the model to get the pooled output
        Then applies a linear layer to get the logits for the entity classification
        Then applies another linear layer to get the logits for the stereotype classification
    """

    def __init__(self, num_labels, num_stp_labels, model_name='xlm-roberta-base'):
        super(TripleClassifier, self).__init__()
        self.model = AutoModel.from_pretrained(model_name)
        self.num_labels = num_labels
        self.num_stp_labels = num_stp_labels
        self.dropout = torch.nn.Dropout(0.1)
        self.linear = torch.nn.Linear(self.model.config.hidden_size, self.num_labels)
        self.stp_linear = torch.nn.Linear(self.model.config.hidden_size, self.num_stp_labels)
        self.softmax = torch.nn.Softmax(dim=1)
        
    
    def forward(self, input_ids, attention_mask):
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs[1]
        pooled_output = self.dropout(pooled_output)
        logits = self.linear(pooled_output)
        stp_logits = self.stp_linear(pooled_output)
        return self.softmax(logits), self.softmax(stp_logits)
    

    def get_entity_loss(self, logits, labels):
        loss_fct = torch.nn.CrossEntropyLoss()
        loss = loss_fct(logits, labels)
        return loss

    def get_stereotype_loss(self, stp_logits, stp_labels):
        """
            stp_logits: (batch_size, num_stp_labels)
            stp_labels: (batch_size, num_stp_labels)
            This method calculates the loss for the stereotype classification such that,
            if stp_labels shape is (batch_size, num_stp_labels), then the loss is calculated using cross entropy loss
            else if stp_labels shape is (batch_size, num_stp_labels, k), then the loss is calculated using binary cross entropy loss
        """

        if len(stp_labels.shape) == 1:
            loss_fct = torch.nn.CrossEntropyLoss()
            loss = loss_fct(stp_logits, stp_labels)
        else:
            loss_fct = torch.nn.BCELoss()
            loss = loss_fct(stp_logits.float(), stp_labels.float())
        return loss
    

    def get_loss(self, logits, stp_logits, labels, stp_labels):
        entity_loss = self.get_entity_loss(logits, labels)
        stp_loss = self.get_stereotype_loss(stp_logits, stp_labels)
        return entity_loss, stp_loss