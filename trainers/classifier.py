import numpy as np
from constants import *
import streamlit as st
import pandas as pd
import torch
from tqdm.auto import tqdm
import os
from torch.utils.tensorboard import SummaryWriter
from stqdm import stqdm


class ClassificationTrainer:
    def __init__(self, model, tokenizer, dataloaders, compute_fn, args):
        
        self.model = model.to(DEVICE)
        self.tokenizer = tokenizer
        self.dataloaders = dataloaders
        if args.phase == TRAINING_PHASE:
            self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=args.lr)
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=args.num_epochs)
        
        self.writer = SummaryWriter(log_dir=args.log_dir)
        self.compute_metrics_fn = compute_fn
        self.models_dir = args.models_dir
        self.results_dir = args.log_dir
        self.logs_dir = args.log_dir
        self.results = list()
        self.results_container = st.empty()
        

    def train_epoch(self, epoch):
        self.model.train()
        epoch_metrics = {f'{TRAIN_LABEL}_{LOSS}': 0}
        for i, batch in stqdm(enumerate(self.dataloaders[TRAIN_LABEL]), desc=f'Batch', total=len(self.dataloaders[TRAIN_LABEL])):
            loss, logits, labels = self.step(batch)
            epoch_metrics[f'{TRAIN_LABEL}_{LOSS}'] += loss.item()
            
            if self.compute_metrics_fn is not None:
                metrics = self.compute_metrics_fn(logits, labels)
                for metric in metrics:
                    metric_label = f'{TRAIN_LABEL}_{metric}'
                    if metric_label not in epoch_metrics:
                        epoch_metrics[metric_label] = 0
                    epoch_metrics[metric_label] += metrics[metric]

            ### Gradient Clipping to prevent exploding gradients
            # torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            break
            loss.backward()
            self.optimizer.step()
            self.scheduler.step()
            self.optimizer.zero_grad()

            if i % 100 == 0:
                print(f'Epoch {epoch + 1} Batch {i+1} Avg Loss: {epoch_metrics[f"{TRAIN_LABEL}_{LOSS}"] / (i + 1)}')
                test_metrics = self.evaluate()
                unseen_metrics = self.evaluate(UNSEEN_LABEL)
                print(f'Test Metrics: ', end='')
                for metric in test_metrics:
                    print(f'{metric}: {test_metrics[metric]:.3f}', end=' ')
                print()
                print(f'Epoch {epoch + 1} Unseen Metrics: ', end='')
                for metric in unseen_metrics:
                    print(f'{metric}: {unseen_metrics[metric]:.3f}', end=' ')
                print()

            

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
        eval_metrics = {f'{split_type}_{LOSS}': 0}
        for batch in tqdm(self.dataloaders[split_type], desc=f'Evaluation'):
            
            loss, logits, labels = self.step(batch)
            eval_metrics[f'{split_type}_{LOSS}'] += loss.item()

            if self.compute_metrics_fn is not None:
                metrics = self.compute_metrics_fn(logits, labels)
                for metric in metrics:
                    metric_label = f'{split_type}_{metric}'
                    if metric_label not in eval_metrics:
                        eval_metrics[metric_label] = 0
                    eval_metrics[metric_label] += metrics[metric]
            
            break
   
        for metric in eval_metrics:
            eval_metrics[metric] /= len(self.dataloaders[split_type])
        
        
        return eval_metrics
    

    def train(self, num_epochs):
        best_loss = float('inf')
        for epoch in stqdm(range(num_epochs), desc='Training Classifer'):
        # for epoch in stqdm(range(num_epochs), desc='Training Classifer'):
            train_metrics = self.train_epoch(epoch)
            self.write_metrics(train_metrics, epoch, TRAIN_LABEL)

            test_metrics = self.evaluate(TEST_LABEL)
            self.write_metrics(test_metrics, epoch, TEST_LABEL)

            unseen_metrics = self.evaluate(UNSEEN_LABEL)
            self.write_metrics(unseen_metrics, epoch, UNSEEN_LABEL)

            
            if test_metrics[f'{TEST_LABEL}_{LOSS}'] < best_loss:
                best_loss = test_metrics[f'{TEST_LABEL}_{LOSS}']
                self.save_model()
                print(f'Best model saved at epoch {epoch}')
            
            metrics = {**{EPOCH: epoch}, **train_metrics, **test_metrics, **unseen_metrics}
            self.results.append(metrics)
            print(f'Epoch {epoch} Metrics: ', end='')
            for metric in metrics:
                print(f'{metric}: {metrics[metric]:.3f}', end=' ')
            print()

            with self.results_container.container():
                st.subheader(f"## Results")
                for split_type, metrics in zip(
                    [TRAIN_LABEL, TEST_LABEL, UNSEEN_LABEL], \
                        [train_metrics, test_metrics, unseen_metrics]
                ):
                    relevant_columns = [metric for metric in metrics if metric.startswith(split_type)]
                    df = pd.DataFrame(self.results)[relevant_columns]
                    df.insert(0, EPOCH, range(1, len(df)+1))
                    
                    st.markdown(f"#### {split_type}")
                    st.dataframe(df, hide_index=True)
            
            break
        
    def save_model(self):
        pass


    def get_recommendations(self, n=5):
        """
            Get top n predictions for each label
        """
        with torch.no_grad():
            recommendations = dict()
            for batch in stqdm(self.dataloaders[TEST_LABEL], desc='Getting Recommendations'):
                _, logits, labels = self.step(batch)
                # print("Total number of labels: ", len(labels))
                logits, labels = logits.cpu().numpy(), labels.cpu().numpy()
                n_predictions = np.argsort(logits, axis=1)[:, -n:]
                for label, predictions in zip(labels, n_predictions):
                    
                    recommendations[label] = predictions.tolist()
            # print(recommendations.keys())
        return recommendations
    

    def save_results(self):
        if not os.path.exists(self.logs_dir):
            os.makedirs(self.logs_dir)
        
        df = pd.DataFrame(self.results)
        df.to_csv(os.path.join(self.logs_dir, 'results.csv'), index=False)
        print(f'Saved results at {self.logs_dir}')

    
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
        
        if not os.path.exists(self.logs_dir):
            os.makedirs(self.logs_dir)
        
        df = pd.DataFrame(self.results)
        df.to_csv(os.path.join(self.logs_dir, 'results.csv'), index=False)
        print(f'Saved results at {self.logs_dir}')