from torch.utils.tensorboard import SummaryWriter
from data_utils import get_recommendation_metrics, get_recommendation_metrics_multi_label
import torch
from tqdm.auto import tqdm
import os
import torch.nn as nn
from transformers import Trainer, TrainingArguments
from transformers import AutoModelForCausalLM
from models import UMLGPT
from data_generation_utils import get_gpt2_dataset
from data_generation_utils import get_dataloaders
from data_generation_utils import get_pretrained_lm_tokenizer, get_word_tokenizer_tokenizer
from data_generation_utils import get_generative_uml_dataset


device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

class LMTrainer:
    def __init__(self, 
            model, 
            train_dataloader, 
            test_dataloader, 
            num_epochs=10, 
            alpha=0.5,
            save_dir='models',
            multi_label=False,
        ):
        self.model = model
        self.model.to(device)

        self.train_dataloader = train_dataloader
        self.test_dataloader = test_dataloader
        self.num_epochs = num_epochs
        self.alpha = alpha
        self.multi_label = multi_label
        self.entity_metric_func = get_recommendation_metrics
        self.spt_metric_func = get_recommendation_metrics_multi_label if multi_label else get_recommendation_metrics
        self.optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=1, gamma=0.9)
        self.writer = SummaryWriter(log_dir='logs')
        self.metrics = ['MRR', 'Hits@1', 'Hits@3', 'Hits@5', 'Hits@10']
        self.save_dir = save_dir
    
    
    def test(self, epoch, dataloader=None):
        if dataloader is None:
            dataloader = self.test_dataloader

        self.model.eval()
        with torch.no_grad():
            test_epoch_loss = 0
            test_results_ep = {k: 0 for k in self.metrics}
            test_results_sp = {k: 0 for k in self.metrics}
            for batch in tqdm(self.test_dataloader, desc="Batches"):
                
                if not batch['entity_mask'].sum() or not batch['super_type_mask'].sum():
                    continue

                loss, entity_logits, entity_labels, spt_logits, super_type_labels = self.step(batch)

                assert len(entity_labels) == len(entity_logits)
                assert len(super_type_labels) == len(spt_logits)


                test_epoch_loss += loss.item()

                test_ep_metrics, test_sp_metrics = self.get_metrics(
                    entity_logits, spt_logits, entity_labels, super_type_labels)
                
                for metric in test_results_ep:
                    test_results_ep[metric] += test_ep_metrics[metric]
                
                for metric in test_results_sp:
                    test_results_sp[metric] += test_sp_metrics[metric]

                # break
            
            avg_test_epoch_loss = test_epoch_loss / len(self.test_dataloader)
            print(f"Test Epoch {epoch} Loss: {avg_test_epoch_loss}")
            self.writer.add_scalar("Test Loss", avg_test_epoch_loss, epoch)


            for metric in test_results_ep:
                test_results_ep[metric] /= len(self.test_dataloader)
                self.writer.add_scalar(f"Test {metric}", test_results_ep[metric], epoch)

            for metric in test_results_sp:
                test_results_sp[metric] /= len(self.test_dataloader)
                self.writer.add_scalar(f"Test {metric}", test_results_sp[metric], epoch)
            

            print("-"*100)
            print("Metrics on Test set")
            print("Entity Prediction", test_results_ep)
            print("Super Type Prediction", test_results_sp)
            print("-"*100)
            print()

            

    def train(self):
        self.model.train()
        min_epoch_loss = float('inf')
        best_metrics_ep = {k: 0 for k in self.metrics}
        best_metrics_sp = {k: 0 for k in self.metrics}
        for epoch in tqdm(range(self.num_epochs), desc="Epochs"):
            epoch_loss = 0
            epoch_results_ep = {k: 0 for k in self.metrics}
            epoch_results_sp = {k: 0 for k in self.metrics}
            
            for batch in tqdm(self.train_dataloader, desc="Batches"):
                # print(batch['input_ids'].shape, batch['attention_mask'].shape, entity_labels.shape, super_type_labels.shape)
                
                if not batch['entity_mask'].sum() or not batch['super_type_mask'].sum():
                    continue
                    
                self.optimizer.zero_grad()
                loss, entity_logits, entity_labels, spt_logits, super_type_labels = self.step(batch)

                assert len(entity_labels) == len(entity_logits)
                assert len(super_type_labels) == len(spt_logits)

                loss.backward()
                self.optimizer.step()
                self.scheduler.step()

                epoch_loss += loss.item()

                train_ep_metrics, train_sp_metrics = self.get_metrics(
                    entity_logits, spt_logits, entity_labels, super_type_labels)

                for metric in epoch_results_ep:
                    epoch_results_ep[metric] += train_ep_metrics[metric]
                
                for metric in epoch_results_sp:
                    epoch_results_sp[metric] += train_sp_metrics[metric]
                
                # break
                
            
            avg_epoch_loss = epoch_loss / len(self.train_dataloader)
            print(f"Epoch {epoch} Loss: {avg_epoch_loss}")
            self.writer.add_scalar("Train Loss", avg_epoch_loss, epoch)


            for metric in epoch_results_ep:
                epoch_results_ep[metric] /= len(self.train_dataloader)
                self.writer.add_scalar(f"Train {metric}", epoch_results_ep[metric], epoch)

            for metric in epoch_results_sp:
                epoch_results_sp[metric] /= len(self.train_dataloader)
                self.writer.add_scalar(f"Train {metric}", epoch_results_sp[metric], epoch)

            if avg_epoch_loss < min_epoch_loss:
                min_epoch_loss = avg_epoch_loss
                torch.save(self.model.state_dict(), self.save_dir + f'/XML4UML_ckpt_{epoch}.pt')

            print("-"*100)
            print("Metrics on train set")
            print("Entity Prediction", epoch_results_ep)
            print("Super Type Prediction", epoch_results_sp)
            print("-"*100)
            print()

            for metric in epoch_results_ep:
                if epoch_results_ep[metric] > best_metrics_ep[metric]:
                    best_metrics_ep[metric] = epoch_results_ep[metric]
            
            for metric in epoch_results_sp:
                if epoch_results_sp[metric] > best_metrics_sp[metric]:
                    best_metrics_sp[metric] = epoch_results_sp[metric]


            if epoch % 2 == 0:
                print("-"*100)
                self.test(epoch)
                print("-"*100)
                
            
            # break
    def step(self, batch):
        
        entity_mask = batch['entity_mask']
        spt_mask = batch['super_type_mask']

        super_type_labels = batch['super_type_label']
        entity_labels = batch['entity_label']

        entity_logits, spt_logits = self.model(batch)

        entity_loss = self.model.get_entity_loss(entity_logits[entity_mask], entity_labels[entity_mask])
        spt_loss = self.model.get_super_type_loss(spt_logits[spt_mask], super_type_labels[spt_mask])
        
        loss = self.alpha * entity_loss + (1 - self.alpha) * spt_loss

        return loss, entity_logits[entity_mask], entity_labels[entity_mask], spt_logits[spt_mask], super_type_labels[spt_mask]
    

    def get_metrics(self, entity_logits, spt_logits, entity_labels, super_type_labels):
        entity_metrics = self.entity_metric_func(entity_logits, entity_labels)
        super_type_metrics = self.spt_metric_func(spt_logits, super_type_labels)
        return entity_metrics, super_type_metrics


class UMLGPTTrainer:
    def __init__(self, model, tokenizer, dataloaders, args):
        self.model = model
        self.lr = args.lr
        self.batch_size = args.batch_size
        self.tokenizer = tokenizer
        self.dataloaders = dataloaders
        self.args = args
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.lr)
        self.criterion = nn.CrossEntropyLoss(ignore_index=-100)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=10, gamma=0.1)
        self.writer = SummaryWriter(log_dir=args.log_dir)
    
    def train(self, epochs):
        self.model.train()
        for epoch in range(epochs):
            epoch_loss = 0
            best_test_loss = float('inf')
            for batch in tqdm(self.dataloaders['train'], desc=f'Epoch {epoch}'):
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['labels'].to(device)
                logits = self.model(input_ids, attention_mask)
                loss = self.criterion(logits.view(-1, logits.size(-1)), labels.view(-1))
                epoch_loss += loss.item()

                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                loss.backward()
                self.optimizer.step()
                self.scheduler.step()
                self.optimizer.zero_grad()

            print(f'Epoch {epoch} Loss: {epoch_loss}')
            self.writer.add_scalar('Loss/train', epoch_loss, epoch)
            # self.scheduler.step()
            if epoch % 10 == 0:
                test_loss = self.evaluate(epoch, 'test')

                if test_loss < best_test_loss:
                    best_test_loss = test_loss
                    self.save_model(f'best_model.pt')
                    print(f'Best model saved at epoch {epoch}')

                
    def evaluate(self, epoch, split_type):
        self.model.eval()
        epoch_loss = 0
        for batch in tqdm(self.dataloaders[split_type], desc=f'Evaluation'):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            logits = self.model(input_ids, attention_mask)
            loss = self.criterion(logits.view(-1, logits.size(-1)), labels.view(-1))
            epoch_loss += loss.item()
        print(f'{split_type} Loss: {epoch_loss}')
        self.writer.add_scalar('Loss/eval', epoch_loss, epoch)
        return epoch_loss


    def save_model(self, file_name):
        models_dir = os.path.join(self.args.log_dir, 'models')
        if not os.path.exists(models_dir):
            os.makedirs(models_dir)
        file_name = os.path.join(models_dir, file_name)
        torch.save(self.model.state_dict(), file_name)



def get_uml_gpt(dataset, tokenizer, args):
    embed_dim = args.embed_dim
    block_size = max([i.shape[-1] for split_type in dataset for i in dataset[split_type][:]['input_ids']])
    n_layer = args.num_layers
    n_head = args.num_heads

    uml_gpt = UMLGPT(len(tokenizer), embed_dim, block_size, n_layer, n_head)
    uml_gpt.to(device)
    return uml_gpt


def train_hugging_face_gpt(data, args):
    model_name = args.gpt_model
    tokenizer = get_pretrained_lm_tokenizer(model_name, special_tokens=args.special_tokens)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    model.resize_token_embeddings(len(tokenizer))
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    model.config.pad_token_id = model.config.eos_token_id

    print('Creating dataset...')
    dataset = get_gpt2_dataset(data, tokenizer)
    print('Done!')

    training_args = TrainingArguments(
        output_dir=args.log_dir,          # output directory
        num_train_epochs=args.num_epochs,              # total number of training epochs
        per_device_train_batch_size=args.batch_size,   # batch size per device during training
        per_device_eval_batch_size=args.batch_size,    # batch size for evaluation
        warmup_steps=500,                # number of warmup steps for learning rate scheduler
        weight_decay=0.01,               # strength of weight decay
        logging_dir=args.log_dir,            # directory for storing logs
        logging_steps=10,
        save_steps=10,
        save_total_limit=1,
        evaluation_strategy='steps',
        eval_steps=10,
        load_best_model_at_end=True,
        metric_for_best_model='eval_loss',
        # fp16=True,
        greater_is_better=False
    )

    trainer = Trainer(
        model=model,                         # the instantiated Transformers model to be trained
        args=training_args,                  # training arguments, defined above
        train_dataset=dataset['train'],         # training dataset
        eval_dataset=dataset['test']             # evaluation dataset
    )

    trainer.train()

    print('Evaluating on test set...')
    trainer.evaluate(dataset['test'])

    print('Evaluating on unseen set...')
    trainer.evaluate(dataset['unseen'])

    trainer.save_model(os.path.join(args.log_dir, f'uml_{model_name}'))


    print('Done!')


def train_umlgpt(dataset, token_type, args):
    if token_type == 'PT':
        print("Creating pretrained LM tokenizer...")
        tokenizer = get_pretrained_lm_tokenizer('bert-base-cased', special_tokens=args.special_tokens)
        print("Done!")
    else:
        print("Creating word tokenizer...")
        tokenizer = get_word_tokenizer_tokenizer(dataset, special_tokens=args.special_tokens)
        print("Done!")
        

    print("Tokenize dataset...")
    tokenized_dataset = get_generative_uml_dataset(dataset, tokenizer)
    print("Done!")

    uml_gpt = get_uml_gpt(tokenized_dataset, tokenizer, args)

    print("Creating dataloaders and trainer...")
    trainer = UMLGPTTrainer(uml_gpt, tokenizer, get_dataloaders(tokenized_dataset), args)
    print("Done!")

    print("Training...")
    trainer.train(args.num_epochs)
    trainer.save_model(f'{token_type}_uml_gpt_{args.num_epochs}.pt')
