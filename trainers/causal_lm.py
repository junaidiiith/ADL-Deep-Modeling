import pandas as pd
from constants import *
import streamlit as st
from tqdm.auto import tqdm
import os

from stqdm import stqdm

class CausalLMTrainer:
    def __init__(self, model, tokenizer, dataset, args):
        self.model = model.to(DEVICE)
        self.tokenizer = tokenizer
        self.dataloaders = {
            split_type: torch.utils.data.DataLoader(
                dataset[split_type], 
                batch_size=args.batch_size, 
                shuffle=args.phase == TRAINING_PHASE,
            ) for split_type in dataset
        }
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=args.lr)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=args.num_epochs)
        self.models_dir = args.models_dir
        self.results_dir = args.log_dir
        self.results = list()
        self.results_container = st.empty()


    def step(self, batch):
        """
            Get Loss and Logits from AutoModelForSequenceClassification model
        """

        input_ids = batch['input_ids'].to(DEVICE)
        attention_mask = batch['attention_mask'].to(DEVICE)
        labels = input_ids.clone().to(DEVICE)
        
        loss = self.model(input_ids, attention_mask=attention_mask, labels=labels)[0]
        return loss


    def train_epoch(self, epoch):
        self.model.train()
        epoch_loss = 0
        for i, batch in stqdm(enumerate(self.dataloaders[TRAIN_LABEL]), desc=f'Epoch {epoch + 1}', total=len(self.dataloaders[TRAIN_LABEL])):
            self.optimizer.zero_grad()
            loss = self.step(batch)
            loss.backward()
            self.optimizer.step()

            epoch_loss += loss.item()

            if i % 100 == 0:
                print(f'Epoch {epoch} Batch {i} Avg Loss: {epoch_loss / (i + 1)}')
                test_loss = self.evaluate(TEST_LABEL)
                unseen_loss = self.evaluate(UNSEEN_LABEL)
                print(f'Epoch {epoch} Test Loss: {test_loss} and Unseen Loss: {unseen_loss}')
            
            break

        return epoch_loss / len(self.dataloaders[TRAIN_LABEL])
    
    
    def evaluate(self, split_type=TEST_LABEL):
        """
            Evaluate the model on the test set
            Args:
                epoch: int
                    The current epoch number
                split_type: str
                    The split type to evaluate on
        """
        self.model.eval()
        test_loss = 0
        for batch in tqdm(self.dataloaders[split_type], desc=f'Evaluation'):
            loss = self.step(batch)
            test_loss += loss.item()

        test_loss /= len(self.dataloaders[split_type])
        return test_loss
            
    
    def train(self, num_epochs):
        results = list()
        best_loss = float('inf')
        for epoch in stqdm(range(num_epochs)):
        # for epoch in tqdm(range(num_epochs)):
            train_loss = self.train_epoch(epoch)
            test_loss = self.evaluate(TEST_LABEL)
            unseen_loss = self.evaluate(UNSEEN_LABEL)

            print(f'Epoch {epoch} Test Loss: {test_loss} and Unseen Loss: {unseen_loss}')
            test_loss = 0
            if test_loss < best_loss:
                best_loss = test_loss
                self.save_model()
                print(f'Best model saved at epoch {epoch}')

            results.append(
                {
                    EPOCH: epoch,
                    "train_loss": train_loss,
                    "test_loss": test_loss,
                    "unseen_loss": unseen_loss,
                }
            )
            break
        
        return results

    
    def save_model(self):
        if not os.path.exists(self.models_dir):
            os.makedirs(self.models_dir)

        self.model.save_pretrained(self.models_dir)
        self.tokenizer.save_pretrained(self.models_dir)
        print(f'Saved model at {self.models_dir}')
    

    def save_results(self):
        if not os.path.exists(self.logs_dir):
            os.makedirs(self.logs_dir)

        df = pd.DataFrame(self.results)
        df.to_csv(os.path.join(self.logs_dir, 'results.csv'), index=False)
        print(f'Saved results at {self.logs_dir}')
