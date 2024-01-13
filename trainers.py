import streamlit as st
import pandas as pd
from torch.utils.tensorboard import SummaryWriter
from transformers import DataCollatorForLanguageModeling
from utils import compute_metrics, compute_auc
import torch
from tqdm.auto import tqdm
from stqdm import stqdm
import os
import torch.nn as nn
from transformers import Trainer, TrainingArguments
from transformers import AutoModelForCausalLM
from transformers import GPT2Config, GPT2ForSequenceClassification
from transformers import AutoModelForSequenceClassification
from models import UMLGPT
from transformers.integrations import NeptuneCallback
from data_generation_utils import get_gpt2_dataset
from data_generation_utils import get_dataloaders
from data_generation_utils import get_pretrained_lm_tokenizer, get_word_tokenizer_tokenizer
from data_generation_utils import get_generative_uml_dataset
import transformers
import itertools
from data_generation_utils import SPECIAL_TOKENS
from constants import *


def suppress_neptune(trainer):
    for cb in trainer.callback_handler.callbacks:
        if isinstance(cb, NeptuneCallback):
            trainer.callback_handler.remove_callback(cb)


def compute_loss(pos_score, neg_score):
    scores = torch.cat([pos_score, neg_score]).to(DEVICE)
    labels = torch.cat([torch.ones(pos_score.shape[0]), torch.zeros(neg_score.shape[0])]).to(DEVICE)
    return torch.nn.BCEWithLogitsLoss()(scores.float(), labels.float())



def get_tokenizer(tokenizer_name, data=None, special_tokens=SPECIAL_TOKENS):
    if data is None:
        print("Creating pretrained LM tokenizer...")
        tokenizer = get_pretrained_lm_tokenizer(tokenizer_name, special_tokens=special_tokens)
        print("Done!")
    else:
        print("Creating word tokenizer...")
        tokenizer = get_word_tokenizer_tokenizer(data)
        print("Done!")
    
    return tokenizer



def get_hf_classification_model(model_name, num_labels, tokenizer):
    """
        Get the hugging face classification model
    """
    if 'gpt2' in model_name:
        model_config = GPT2Config.from_pretrained(pretrained_model_name_or_path=model_name, num_labels=num_labels)
        tokenizer.padding_side = "left"
        model = GPT2ForSequenceClassification.from_pretrained(pretrained_model_name_or_path=model_name, config=model_config)
        model.resize_token_embeddings(len(tokenizer)) 
        model.config.pad_token_id = model.config.eos_token_id
    else:
        model = AutoModelForSequenceClassification.from_pretrained(
            pretrained_model_name_or_path=model_name, num_labels=num_labels, ignore_mismatched_sizes=True)
        model.resize_token_embeddings(len(tokenizer))
        
    return model


class UMLGPTTrainer:
    """
        Trainer class for UMLGPT
        This class is used to train the UMLGPT model
        The model is trained for the next token prediction or node (super type or entity) classification task
        In case of token classification, a callable function ``compute_metrics_fn`` is provided
        This function is used to compute the metrics for the task
    """
    def __init__(self, model, dataloaders, args, compute_metrics_fn=None):
        self.model = model
        self.model.to(DEVICE)
        self.lr = args.lr
        self.batch_size = args.batch_size
        self.dataloaders = dataloaders
        self.args = args
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.lr)
        self.criterion = nn.CrossEntropyLoss(ignore_index=-100)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=args.num_epochs)
        self.writer = SummaryWriter(log_dir=args.log_dir)
        self.models_dir = args.models_dir
        self.logs_dir = args.log_dir
        self.compute_metrics_fn = compute_metrics_fn
        self.results = list()
        self.results_container = st.empty()
        
    
    def train(self, epochs):
        self.model.train()
                    
        for epoch in stqdm(range(epochs), desc='Training GPT'):
            best_test_loss = float('inf')
            loss_label = f'{TRAIN_LABEL}_Loss'
            epoch_metrics = {EPOCH: int(f'{epoch+1}'), loss_label: 0}
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

            test_metrics = self.evaluate(epoch, TEST_LABEL)
            unseen_metrics = self.evaluate(epoch, UNSEEN_LABEL)
            
            test_loss = test_metrics[f'{TEST_LABEL}_Loss']
            if test_loss < best_test_loss:
                best_test_loss = test_loss
                self.save_model(f'best_model.pt')
                print(f'Best model saved at epoch {epoch}')
                # with st.empty():
                #     st.write(f'Best model saved at location: {os.path.join(self.models_dir, "best_model.pt")}')
            
            self.results.append({**epoch_metrics, **test_metrics, **unseen_metrics})
            with self.results_container.container():
                st.subheader(f"## Results")
                st.dataframe(pd.DataFrame(self.results))
            

                
    def evaluate(self, epoch, split_type=TEST_LABEL):
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

        self.write_metrics(eval_metrics, epoch, split_type)
        return eval_metrics
    

    def step(self, batch):
        input_ids = batch['input_ids'].to(DEVICE)
        attention_mask = batch['attention_mask'].to(DEVICE)
        labels = batch['labels'].to(DEVICE)
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
        # self.results_placeholders = {
        #     metric: st.empty() for metric in self.results.columns
        #     if metric not in [EPOCH]
        # }

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

            with self.st_results.container():
                st.subheader(f"## Results")
                st.dataframe(pd.DataFrame(self.results), hide_index=True)


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


def get_uml_gpt(vocab_size, args):
    """
        Get the UMLGPT model
        Args:
            input_dim: int
                The input dimension of the model
            args: Namespace
                The arguments
    """

    
    if args.from_pretrained is not None:
        uml_gpt = UMLGPT.from_pretrained(args.from_pretrained)
        print(f'Loaded pretrained model from {args.from_pretrained}')
    else:
        embed_dim = args.embed_dim
        n_layer = args.num_layers
        n_head = args.num_heads
        block_size = args.block_size
        uml_gpt = UMLGPT(vocab_size, embed_dim, block_size, n_layer, n_head)
    
    uml_gpt.to(DEVICE)
    return uml_gpt



def train_umlgpt(dataset, args):
    """
        Train the UMLGPT model
        Args:
            dataset: dict
                The dataset dictionary
            args: Namespace
                The arguments
    """
    if args.tokenizer != WORD_TOKENIZER:
        tokenizer = get_tokenizer(args.tokenizer)
    else:
        tokenizer = get_tokenizer(WORD_TOKENIZER, dataset)
        tokenizer.save_pretrained(args.models_dir)
        print(f"Saved tokenizer at {args.models_dir}")

    print("Tokenize dataset...")
    tokenized_dataset = get_generative_uml_dataset(dataset, tokenizer)
    print("Done!")

    uml_gpt = get_uml_gpt(len(tokenizer), args)

    print("Model initialized! with parameters:")
    print(uml_gpt)

    print("Creating dataloaders and trainer...")
    trainer = UMLGPTTrainer(uml_gpt, get_dataloaders(tokenized_dataset, args.batch_size), args)
    print("Done!")

    print("Training...")
    trainer.train(args.num_epochs)
    trainer.save_model(f'final_model.pt')

    ## Create zipfile for args.models_dir
    # print("Creating zip file...")
    # with st.spinner("Saving best model..."):
    #     shutil.make_archive(args.models_dir, 'zip', args.models_dir)


def train_hugging_face_gpt(data, args):
    """
        Train the hugging face GPT model
        Args:
            data: dict
                The data dictionary
            args: Namespace
                The arguments
    """

    results = dict()
    model_name = args.gpt_model
    tokenizer = get_pretrained_lm_tokenizer(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    model.resize_token_embeddings(len(tokenizer))
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    model.config.pad_token_id = model.config.eos_token_id

    print('Creating dataset...')
    dataset = get_gpt2_dataset(data, tokenizer)
    print('Done!')

    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False  # Set to True if you want to perform masked language modeling
    )


    training_args = TrainingArguments(
        output_dir=args.log_dir,          # output directory
        num_train_epochs=args.num_epochs,              # total number of training epochs
        per_device_train_batch_size=args.batch_size,   # batch size per device during training
        per_device_eval_batch_size=args.batch_size,    # batch size for evaluation
        warmup_steps=500,                # number of warmup steps for learning rate scheduler
        weight_decay=0.01,               # strength of weight decay
        logging_dir=args.log_dir,            # directory for storing logs
        logging_steps=10,
        save_steps=1000,
        save_total_limit=1,
        evaluation_strategy='steps',
        eval_steps=100,
        lr_scheduler_type="cosine",
        load_best_model_at_end=True,
        metric_for_best_model='eval_loss',
        # fp16=True,
        greater_is_better=False
    )

    trainer = Trainer(
        model=model,                         # the instantiated Transformers model to be trained
        args=training_args,                  # training arguments, defined above
        train_dataset=dataset[TRAIN_LABEL],         # training dataset
        eval_dataset=dataset[TEST_LABEL],          # evaluation dataset
        data_collator=data_collator,
        tokenizer=tokenizer,
    )

    suppress_neptune(trainer)
    st_results = st.empty()
    results = list()
    best_loss = float('inf')
    for epoch in stqdm(range(args.num_epochs), desc="Training Epoch: "):
        # trainer.train()
        # print("Evaluating on test set...")
        # test_results = trainer.evaluate(dataset[TEST_LABEL])
        # print(test_results)

        # print("Evaluating on unseen set...")
        # unseen_results = trainer.evaluate(dataset[UNSEEN_LABEL])
        # print(unseen_results)

        test_results = {'eval_loss': 0, 'eval_accuracy': 0}
        unseen_results = {'eval_loss': 0, 'eval_accuracy': 0}
        
        if test_results['eval_loss'] < best_loss:
            best_loss = test_results['eval_loss']
            with st.spinner("Saving best model..."):
                trainer.save_model(args.models_dir)
        
        
        test_results = {f"test_{k}": v for k, v in test_results.items()}
        unseen_results = {f"unseen_{k}": v for k, v in unseen_results.items()}
        results.append({**{'Epoch': epoch}, **test_results, **unseen_results})

        with st_results.container():
            st.markdown(f"### {epoch+1} Results")
            st.dataframe(pd.DataFrame(results), hide_index=True)
    
    with st_results.container():
        st.markdown(f"## Pretraining Results")
        st.dataframe(pd.DataFrame(results), hide_index=True)


def train_hf_for_classification(dataset, tokenizer, args):
    """
        Train the hugging face classification model
        Args:
            dataset: dict
                The dataset dictionary
            tokenizer: PreTrainedTokenizer
                The tokenizer
            args: Namespace
                The arguments
    """
    model_name = args.from_pretrained
    batch_size = args.batch_size
    train, test, unseen = dataset[TRAIN_LABEL], dataset[TEST_LABEL], dataset[UNSEEN_LABEL]
    # Show the training loss with every epoch
    logging_steps = 100
    print(f"Using model...{model_name}")
    model = get_hf_classification_model(model_name, dataset[TRAIN_LABEL].num_classes, tokenizer)
    model.resize_token_embeddings(len(tokenizer))
    print("Finetuning model...")
    training_args = TrainingArguments(
        output_dir=args.models_dir,
        overwrite_output_dir=True,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        learning_rate=2e-5,
        weight_decay=0.01,
        warmup_steps=100,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        # fp16=True,
        logging_steps=logging_steps,
        num_train_epochs=1,
        save_total_limit=2,
        load_best_model_at_end=True,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train,
        eval_dataset=test,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics
    )
    for cb in trainer.callback_handler.callbacks:
        if isinstance(cb, transformers.integrations.NeptuneCallback):
            trainer.callback_handler.remove_callback(cb)

    best_loss = float('inf')
    results = list()
    st_results = st.empty()
    for epoch in stqdm(range(args.num_epochs), desc="Training Epoch: "):
        # trainer.train()
        # print("Evaluating on test set...")
        # test_results = trainer.evaluate(test)
        # print(test_results)

        # print("Evaluating on unseen set...")
        # unseen_results = trainer.evaluate(unseen)
        # print(unseen_results)

        test_results = {'eval_loss': 0, 'accuracy': 0}
        unseen_results = {'eval_loss': 0, 'accuracy': 0}

        if test_results['eval_loss'] < best_loss:
            best_loss = test_results['eval_loss']
            with st.spinner("Saving best model ..."):
                trainer.save_model(args.models_dir)
            
        
        test_results = {f"test_{k}": v for k, v in test_results.items()}
        unseen_results = {f"unseen_{k}": v for k, v in unseen_results.items()}
        results.append({**{'Epoch': epoch}, **test_results, **unseen_results})

        with st_results.container():
            st.markdown(f"### Epoch {epoch+1} Results")
            st.dataframe(pd.DataFrame(results), hide_index=True)
    
    with st_results.container():
        st.markdown(f"### HF Classification {epoch+1} Results")
        st.dataframe(pd.DataFrame(results), hide_index=True)