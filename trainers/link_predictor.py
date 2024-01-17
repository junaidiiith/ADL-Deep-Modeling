import os
import itertools
import torch
import pandas as pd
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
import streamlit as st
from metrics import compute_auc, compute_loss
from stqdm import stqdm



from constants import *

class GNNLinkPredictionTrainer:
    """
        Trainer class for GNN Link Prediction
        This class is used to train the GNN model for the link prediction task
        The model is trained to predict the link between two nodes
    """
    def __init__(self, model, predictor, args) -> None:
        self.embedding_model_name = args.embedding_model
        self.model = model
        self.predictor = predictor
        self.model.to(DEVICE)
        self.predictor.to(DEVICE)

        if args.phase == TRAINING_PHASE:
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
            metric: st.empty() for metric in self.results
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
                print(f"Train Loss: {epoch_loss / (i + 1)} and Train Accuracy: {epoch_acc / (i + 1)}")
        

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
        for epoch in tqdm(range(num_epochs), desc="Running Epochs"):
        # for epoch in stqdm(range(num_epochs), desc="Running Epochs"):
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
                self.save_model()
            
            # self.scheduler.step()
            self.write_results(outputs)
            self.results.append({EPOCH: epoch+1, TRAIN_LOSS: train_loss, TEST_LOSS: test_loss, TEST_ACC: test_acc})

            with self.st_results.container():
                st.subheader(f"## Results")
                st.dataframe(pd.DataFrame(self.results), hide_index=True)


        print(f"Accuracy: {max_val_acc}")
        df = pd.DataFrame(self.results)
        max_output = dict(df.loc[df[TEST_ACC].idxmax(axis=0)])
        
        return max_output


    def save_model(self):
        pth = os.path.join(self.models_dir, self.embedding_model_name)
        if not os.path.exists(pth):
            os.makedirs(pth)
        
        self.model.save_pretrained(pth)
        self.predictor.save_pretrained(pth)
        
        print(f'Saved model at {pth}')
    


    def write_results(self, outputs):
        with open(os.path.join(self.logs_dir, 'results.txt'), 'a') as f:
            for output in outputs:
                f.write(f"Epoch {output[EPOCH]} Train Loss: {output[TRAIN_LOSS]} and Test Loss: {output[TEST_LOSS]} and Test Accuracy: {output[TEST_ACC]}\n")

        df = pd.DataFrame(self.results)
        df.to_csv(os.path.join(self.logs_dir, 'results.csv'), index=False)