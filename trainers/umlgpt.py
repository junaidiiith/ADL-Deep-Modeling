import os
from constants import *
from trainers.classifier import ClassificationTrainer


class UMLGPTTrainer(ClassificationTrainer):
    """
        Trainer class for UMLGPT
        This class is used to train the UMLGPT model
        The model is trained for the next token prediction or node (super type or entity) classification task
        In case of token classification, a callable function ``compute_metrics_fn`` is provided
        This function is used to compute the metrics for the task
    """
    def __init__(self, model, tokenizer, dataloaders, args, compute_metrics_fn=None):
        super().__init__(model, tokenizer, dataloaders, compute_metrics_fn, args)
        
    

    def step(self, batch):
        input_ids = batch['input_ids'].to(DEVICE)
        attention_mask = batch['attention_mask'].to(DEVICE)
        labels = batch['labels'].to(DEVICE)

        # print(input_ids.shape, attention_mask.shape)
        logits = self.model(input_ids, attention_mask)
        
        loss = self.model.get_loss(logits, labels)
        return loss, logits, labels


    def save_model(self):
        if not os.path.exists(self.models_dir):
            os.makedirs(self.models_dir)

        file_name = os.path.join(self.models_dir, BEST_MODEL_LABEL)
        torch.save(self.model.state_dict(), file_name)
        self.tokenizer.save_pretrained(self.models_dir)
        print(f'Saved model at {file_name}')
        