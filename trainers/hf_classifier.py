from constants import *
import os
from trainers.classifier import ClassificationTrainer


class HFClassificationTrainer(ClassificationTrainer):
    def __init__(self, model, tokenizer, dataloaders, compute_fn, args):
        
        super().__init__(model, tokenizer, dataloaders, compute_fn, args)
        


    def step(self, batch):
        """
            Get Loss and Logits from AutoModelForSequenceClassification model
        """
        
        input_ids = batch['input_ids'].to(DEVICE)
        attention_mask = batch['attention_mask'].to(DEVICE)
        labels = batch['labels'].to(DEVICE)

        # print(input_ids.shape)

        outputs = self.model(input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        logits = outputs.logits

        return loss, logits, labels


    def save_model(self):
        if not os.path.exists(self.models_dir):
            os.makedirs(self.models_dir)

        self.model.save_pretrained(self.models_dir)
        self.tokenizer.save_pretrained(self.models_dir)
        print(f'Saved model at {self.models_dir}')
