import streamlit as st
from stqdm import stqdm
import pandas as pd
import os
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from constants import *


class UMLGPTTrainer:
    """
        Trainer class for UMLGPT
        This class is used to train the UMLGPT model
        The model is trained for the next token prediction or node (super type or entity) classification task
        In case of token classification, a callable function ``compute_metrics_fn`` is provided
        This function is used to compute the metrics for the task
    """
    def __init__(self, model, dataloaders, args, compute_metrics_fn=None):
        self.args = args
        self.dataloaders = dataloaders
        self.model = model
        self.model.to(DEVICE)
        if args.phase == TRAINING_PHASE:
            self.lr = args.lr
            self.batch_size = args.batch_size
            self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.lr)
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=args.num_epochs)
        
        self.criterion = nn.CrossEntropyLoss(ignore_index=-100)
    
        self.writer = SummaryWriter(log_dir=args.log_dir)
        self.models_dir = args.models_dir
        self.logs_dir = args.log_dir

        self.compute_metrics_fn = compute_metrics_fn
        self.results = list()
        self.results_container = st.empty()
    

    def step(self, batch):
        input_ids = batch['input_ids'].to(DEVICE)
        attention_mask = batch['attention_mask'].to(DEVICE)
        labels = batch['labels'].to(DEVICE)

        # print(input_ids.shape, attention_mask.shape)
        logits = self.model(input_ids, attention_mask)
        
        loss = self.model.get_loss(logits, labels)
        return loss, logits, labels


    def train_epoch(self, epoch):
        self.model.train()
        loss_label = f'{TRAIN_LABEL}_loss'
        epoch_metrics = {loss_label: 0}
        # for i, batch in tqdm(enumerate(self.dataloaders[TRAIN_LABEL]), desc=f'Epoch {epoch + 1}', total=len(self.dataloaders[TRAIN_LABEL])):
        for i, batch in stqdm(enumerate(self.dataloaders[TRAIN_LABEL]), desc=f'Epoch {epoch + 1}', total=len(self.dataloaders[TRAIN_LABEL])):
            loss, _, _ = self.step(batch)
            epoch_metrics[loss_label] += loss.item()

            # self.add_metrics(epoch_metrics, logits, labels, TRAIN_LABEL)
            

            ### Gradient Clipping to prevent exploding gradients
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            loss.backward()
            self.optimizer.step()
            # self.scheduler.step()
            self.optimizer.zero_grad()

            if i % 100 == 0:
                print(f'Epoch {epoch} Batch {i} Avg Loss: {epoch_metrics[loss_label] / (i + 1)}')

            break

        self.scheduler.step()
        for metric in epoch_metrics:
            epoch_metrics[metric] /= len(self.dataloaders[TRAIN_LABEL])
        
        return epoch_metrics


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
        loss_label = f'{split_type}_loss'
        eval_metrics = {loss_label: 0}
        # for batch in tqdm(self.dataloaders[split_type], desc=f'Evaluation'):
        for batch in stqdm(self.dataloaders[split_type], desc=f'Evaluation'):
            loss, logits, labels = self.step(batch)
            eval_metrics[loss_label] += loss.item()
            
            self.add_metrics(eval_metrics, logits, labels, split_type)
            

        for metric in eval_metrics:
            eval_metrics[metric] /= len(self.dataloaders[split_type])
        

        return eval_metrics


    def train(self, epochs):
        best_test_loss = float('inf')
        for epoch in stqdm(range(epochs), desc='Training GPT'):
            train_metrics = self.train_epoch(epoch)
            self.write_metrics(train_metrics, epoch, TRAIN_LABEL)

            test_metrics = self.evaluate()
            self.write_metrics(test_metrics, epoch, TEST_LABEL)

            unseen_metrics = self.evaluate(UNSEEN_LABEL)
            self.write_metrics(unseen_metrics, epoch, UNSEEN_LABEL)
            
            test_loss = test_metrics[f'{TEST_LABEL}_{LOSS}']
            if test_loss < best_test_loss:
                best_test_loss = test_loss
                self.save_model(f'best_model.pt')
                print(f'Best model saved at epoch {epoch}')
            
            self.results.append({**train_metrics, **test_metrics, **unseen_metrics})            
            
            with self.results_container.container():
                st.subheader(f"## Results")
                df = pd.DataFrame(self.results)
                df.insert(0, EPOCH, range(1, len(df)+1))
                st.dataframe(df, width=1000, hide_index=True)


    def add_metrics(self, eval_metrics, logits, labels, split_type):
        
        if self.compute_metrics_fn is not None:
            metrics = self.compute_metrics_fn(logits, labels)
            for metric in metrics:
                metric_label = f'{split_type}_{metric}'
                if metric not in eval_metrics:
                    eval_metrics[metric_label] = 0
                eval_metrics[metric_label] += metrics[metric]
        
        

    def save_model(self, file_name):
        if not os.path.exists(self.models_dir):
            os.makedirs(self.models_dir)

        file_name = os.path.join(self.models_dir, file_name)
        torch.save(self.model.state_dict(), file_name)
        print(f'Saved model at {file_name}')
    

    def load_model(self, file_name):
        file_name = os.path.join(self.models_dir, file_name)
        self.model.load_state_dict(torch.load(file_name))
        print(f'Loaded model from {file_name}')
    
    
    def write_metrics(self, metrics, epoch, split_type):
        print(f'Epoch {epoch} {split_type} metrics: ', end='')
        for metric in metrics:
            self.writer.add_scalar(f'Metrics/{split_type}{metric}', metrics[metric], epoch)
            print(f'{metric}: {metrics[metric]:.3f}', end=' ')
        print()

        with open(os.path.join(self.logs_dir, 'results.txt'), 'a') as f:
            f.write(f'Epoch {epoch} {split_type} metrics: ')
            for metric in metrics:
                f.write(f'{metric}: {metrics[metric]:.3f} ')
            f.write('\n')
