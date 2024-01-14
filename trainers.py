from torch.utils.tensorboard import SummaryWriter
from constants import *
import streamlit as st
import pandas as pd
from metrics import compute_auc
import torch
from tqdm.auto import tqdm
from stqdm import stqdm
import os
import torch.nn as nn
import itertools
from training_utils import compute_loss


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
        
    
    def train(self, epochs):
        self.model.train()

        # for epoch in tqdm(range(epochs), desc='Training GPT'):            
        for epoch in stqdm(range(epochs), desc='Training GPT'):
            best_test_loss = float('inf')
            loss_label = f'{TRAIN_LABEL}_Loss'
            epoch_metrics = {EPOCH: int(f'{epoch+1}'), loss_label: 0}
            # for i, batch in tqdm(enumerate(self.dataloaders[TRAIN_LABEL]), desc=f'Epoch {epoch + 1}', total=len(self.dataloaders[TRAIN_LABEL])):
            for i, batch in stqdm(enumerate(self.dataloaders[TRAIN_LABEL]), desc=f'Epoch {epoch + 1}', total=len(self.dataloaders[TRAIN_LABEL])):
                loss, logits, labels = self.step(batch)
                epoch_metrics[loss_label] += loss.item()

                if self.compute_metrics_fn is not None:
                    metrics = self.compute_metrics_fn(logits, labels)
                    for metric in metrics:
                        if metric not in epoch_metrics:
                            epoch_metrics[metric] = 0
                        epoch_metrics[metric] += metrics[metric]
                

                ### Gradient Clipping
                # This is to prevent exploding gradients
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
            
            self.write_metrics(epoch_metrics, epoch, TRAIN_LABEL)

            test_metrics = self.evaluate()
            self.write_metrics(test_metrics, epoch, TEST_LABEL)

            unseen_metrics = self.evaluate(UNSEEN_LABEL)
            self.write_metrics(unseen_metrics, epoch, UNSEEN_LABEL)
            
            test_loss = test_metrics[f'{TEST_LABEL}_Loss']
            if test_loss < best_test_loss:
                best_test_loss = test_loss
                self.save_model(f'best_model.pt')
                print(f'Best model saved at epoch {epoch}')
            
            self.results.append({**epoch_metrics, **test_metrics, **unseen_metrics})
            
            with self.results_container.container():
                st.subheader(f"## Results")
                st.dataframe(pd.DataFrame(self.results))
            

                
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
        loss_label = f'{split_type}_Loss'
        eval_metrics = {loss_label: 0}
        # for batch in tqdm(self.dataloaders[split_type], desc=f'Evaluation'):
        for batch in stqdm(self.dataloaders[split_type], desc=f'Evaluation'):
            loss, logits, labels = self.step(batch)

            if self.compute_metrics_fn is not None:
                metrics = self.compute_metrics_fn(logits, labels)
                for metric in metrics:
                    if metric not in eval_metrics:
                        eval_metrics[metric] = 0
                    eval_metrics[metric] += metrics[metric]

            eval_metrics[loss_label] += loss.item()

        # print("EVM:", eval_metrics)
        for metric in eval_metrics:
            eval_metrics[metric] /= len(self.dataloaders[split_type])
        # print("EVM:", eval_metrics)

        return eval_metrics
    

    def step(self, batch):
        input_ids = batch['input_ids'].to(DEVICE)
        attention_mask = batch['attention_mask'].to(DEVICE)
        labels = batch['labels'].to(DEVICE)

        # print(input_ids.shape, attention_mask.shape)
        logits = self.model(input_ids, attention_mask)
        
        loss = self.model.get_loss(logits, labels)
        return loss, logits, labels


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
                
            

class GNNLinkPredictionTrainer:
    """
        Trainer class for GNN Link Prediction
        This class is used to train the GNN model for the link prediction task
        The model is trained to predict the link between two nodes
    """
    def __init__(self, model, predictor, args) -> None:
        self.model = model
        self.predictor = predictor
        self.model.to(DEVICE)
        self.predictor.to(DEVICE)
        self.optimizer = torch.optim.Adam(itertools.chain(model.parameters(), predictor.parameters()), lr=args.lr)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=args.num_epochs)

        self.models_dir = args.models_dir
        self.logs_dir = args.log_dir
        self.writer = SummaryWriter(log_dir=args.log_dir)

        
        self.edge2index = lambda g: torch.stack(list(g.edges())).contiguous()
        self.args = args
        self.results = list()
        self.st_results = st.empty()
        self.results_placeholders = {
            metric: st.empty() for metric in self.results.columns
            if metric not in [EPOCH]
        }

        print("GNN Trainer initialized.")

    def train(self, dataloader):
        self.model.train()
        self.predictor.train()

        epoch_loss, epoch_acc = 0, 0
        
        for i, batch in tqdm(enumerate(dataloader), desc=f"Training batches", total=len(dataloader)):
            self.optimizer.zero_grad()
            self.model.zero_grad()
            self.predictor.zero_grad()
            
            h = self.get_logits(batch['train_g'])

            pos_score = self.get_prediction_score(batch['train_pos_g'], h)
            neg_score = self.get_prediction_score(batch['train_neg_g'], h)
            loss = compute_loss(pos_score, neg_score)

            loss.backward()
            self.optimizer.step()

            epoch_loss += loss.item()
            epoch_acc += compute_auc(pos_score, neg_score)

            if i % 500 == 0:
                print(f"Epoch {i+1} Train Loss: {epoch_loss / (i + 1)} and Train Accuracy: {epoch_acc / (i + 1)}")
        

        epoch_loss /= len(dataloader)
        epoch_acc /= len(dataloader)
        print(f"Epoch Train Loss: {epoch_loss} and Train Accuracy: {epoch_acc}")
        return epoch_loss, epoch_acc
    

    def test(self, dataloader):
        self.model.eval()
        self.predictor.eval()
        with torch.no_grad():
            epoch_loss, epoch_acc = 0, 0
            for _, batch in tqdm(enumerate(dataloader), desc=f"Evaluating batches", total=len(dataloader)):
                h = self.get_logits(batch['train_g'])

                pos_score = self.get_prediction_score(batch['test_pos_g'], h)
                neg_score = self.get_prediction_score(batch['test_neg_g'], h)
                loss = compute_loss(pos_score, neg_score)

                epoch_loss += loss.item()
                epoch_acc += compute_auc(pos_score, neg_score)

            epoch_loss /= len(dataloader)
            epoch_acc /= len(dataloader)
            print(f"Epoch Test Loss: {epoch_loss} and Test Accuracy: {epoch_acc}")
            return epoch_loss, epoch_acc


    def get_logits(self, g):
        edge_index = self.edge2index(g).to(DEVICE)
        x = g.ndata['h'].float().to(DEVICE)
        h = self.model(x, edge_index)
        return h
    

    def get_prediction_score(self, g, h):
        h = h.to(DEVICE)
        edge_index = self.edge2index(g).to(DEVICE)
        prediction_score = self.predictor(h, edge_index)
        return prediction_score


    def run_epochs(self, dataloader, num_epochs):
        max_val_acc = 0
        outputs = list()
        # for epoch in tqdm(range(num_epochs), desc="Running Epochs"):
        for epoch in stqdm(range(num_epochs), desc="Running Epochs"):
        # for epoch in range(num_epochs):
            train_loss, train_acc = self.train(dataloader)
            self.writer.add_scalar(f"Metrics/TrainLoss", train_loss, epoch)
            self.writer.add_scalar(f"Metrics/TrainAccuracy", train_acc, epoch)

            
            if epoch % 10 == 0:
                print(f"Epoch {epoch} Train Loss: {train_loss}")
            
            test_loss, test_acc = self.test(dataloader)
            self.writer.add_scalar(f"Metrics/TestLoss", test_loss, epoch)
            self.writer.add_scalar(f"Metrics/TestAccuracy", test_acc, epoch)


            if test_acc > max_val_acc:
                max_val_acc = test_acc
                self.save_model(f'best_model.pt')
            
            # self.scheduler.step()
            self.write_results(outputs)
            self.results.append({EPOCH: epoch, TRAIN_LOSS: train_loss, TEST_LOSS: test_loss, TEST_ACC: test_acc})

            # with self.st_results.container():
            #     st.subheader(f"## Results")
            #     st.dataframe(pd.DataFrame(self.results), hide_index=True)


        print(f"Accuracy: {max_val_acc}")
        df = pd.DataFrame(self.results)
        max_output = dict(df.loc[df[TEST_ACC].idxmax(axis=0)])
        
        return max_output

    def save_model(self, file_name):
        if not os.path.exists(self.models_dir):
            os.makedirs(self.models_dir)

        file_name = os.path.join(self.models_dir, file_name)
        torch.save(self.model.state_dict(), file_name)
        # print(f'Saved model at {file_name}')

    def write_results(self, outputs):
        with open(os.path.join(self.logs_dir, 'results.txt'), 'a') as f:
            for output in outputs:
                f.write(f"Epoch {output[EPOCH]} Train Loss: {output[TRAIN_LOSS]} and Test Loss: {output[TEST_LOSS]} and Test Accuracy: {output[TEST_ACC]}\n")
