from torch.utils.tensorboard import SummaryWriter
from transformers import DataCollatorForLanguageModeling
from utils import get_recommendation_metrics, get_recommendation_metrics_multi_label
from utils import compute_metrics
import torch
from tqdm.auto import tqdm
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


device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

def suppress_neptune(trainer):
    for cb in trainer.callback_handler.callbacks:
        if isinstance(cb, NeptuneCallback):
            trainer.callback_handler.remove_callback(cb)


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
    def __init__(self, model, dataloaders, args, compute_metrics_fn=None):
        self.model = model
        self.lr = args.lr
        self.batch_size = args.batch_size
        self.dataloaders = dataloaders
        self.args = args
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.lr)
        self.criterion = nn.CrossEntropyLoss(ignore_index=-100)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=args.num_epochs)
        self.writer = SummaryWriter(log_dir=args.log_dir)
        self.models_dir = args.models_dir
        self.model_str = self.model_str = f'{self.model._get_name()}_Vocab{args.trainer}'
        self.compute_metrics_fn = compute_metrics_fn
    
    def train(self, epochs):
        self.model.train()
                    
        for epoch in range(epochs):
            epoch_loss = 0
            best_test_loss = float('inf')
            epoch_metrics = {'loss': 0}
            for i, batch in tqdm(enumerate(self.dataloaders['train']), desc=f'Epoch {epoch}', total=len(self.dataloaders['train'])):
                loss, logits, labels = self.step(batch)
                epoch_loss += loss.item()

                epoch_metrics['loss'] += epoch_loss

                if self.compute_metrics_fn is not None:
                    metrics = self.compute_metrics_fn(logits, labels)
                    for metric in metrics:
                        if metric not in epoch_metrics:
                            epoch_metrics[metric] = 0
                        epoch_metrics[metric] += metrics[metric]

                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                loss.backward()
                self.optimizer.step()
                # self.scheduler.step()
                self.optimizer.zero_grad()

                if i % 100 == 0:
                    print(f'Epoch {epoch} Batch {i} Avg Loss: {epoch_loss / (i + 1)}')

            self.scheduler.step()

            self.write_metrics(epoch_metrics, epoch, 'train')

            test_loss = self.evaluate(epoch, 'test')
            self.evaluate(epoch, 'unseen')

            if test_loss < best_test_loss:
                best_test_loss = test_loss
                self.save_model(f'{self.model_str}_best_model.pt')
                print(f'Best model saved at epoch {epoch}')
    
                
    def evaluate(self, epoch, split_type='test'):
        self.model.eval()
        eval_metrics = {'loss': 0}
        for batch in tqdm(self.dataloaders[split_type], desc=f'Evaluation'):
            loss, logits, labels = self.step(batch)

            if self.compute_metrics_fn is not None:
                metrics = self.compute_metrics_fn(logits, labels)
                for metric in metrics:
                    if metric not in eval_metrics:
                        eval_metrics[metric] = 0
                    eval_metrics[metric] += metrics[metric]

            eval_metrics['loss'] += loss.item()

        for metric in eval_metrics:
            if metric != 'loss':
                eval_metrics[metric] /= len(self.dataloaders[split_type])

        self.write_metrics(eval_metrics, epoch, split_type)
        return eval_metrics['loss']
    

    def step(self, batch):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
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



def get_uml_gpt(input_dim, args):
    embed_dim = args.embed_dim
    n_layer = args.num_layers
    n_head = args.num_heads
    block_size = args.embed_dim // args.num_heads

    uml_gpt = UMLGPT(input_dim, embed_dim, block_size, n_layer, n_head)
    if args.from_pretrained is not None:
        uml_gpt.load_state_dict(torch.load(os.path.join(args.models_dir, args.from_pretrained)))
        print(f'Loaded pretrained model from {args.from_pretrained}')
    
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
        fp16=True,
        greater_is_better=False
    )

    trainer = Trainer(
        model=model,                         # the instantiated Transformers model to be trained
        args=training_args,                  # training arguments, defined above
        train_dataset=dataset['train'],         # training dataset
        eval_dataset=dataset['test'],          # evaluation dataset
        data_collator=data_collator,
    )

    suppress_neptune(trainer)

    trainer.train()

    print('Evaluating on test set...')
    trainer.evaluate(dataset['test'])

    print('Evaluating on unseen set...')
    trainer.evaluate(dataset['unseen'])

    trainer.save_model(os.path.join(args.log_dir, f'uml_{model_name}'))


    print('Done!')


def get_tokenizer(data, args):
    if args.trainer in ['HFGPT', 'PT']:
        print("Creating pretrained LM tokenizer...")
        tokenizer = get_pretrained_lm_tokenizer(args.model_name, special_tokens=args.special_tokens)
        print("Done!")
    else:
        print("Creating word tokenizer...")
        tokenizer = get_word_tokenizer_tokenizer(data, special_tokens=args.special_tokens)
        print("Done!")
    
    return tokenizer


def train_umlgpt(dataset, args):
    tokenizer = get_tokenizer(dataset, args)
        
    print("Tokenize dataset...")
    tokenized_dataset = get_generative_uml_dataset(dataset, tokenizer)
    print("Done!")

    uml_gpt = get_uml_gpt(tokenized_dataset, tokenizer, args)

    print("Model initialized! with parameters:")
    print(uml_gpt)

    print("Creating dataloaders and trainer...")
    trainer = UMLGPTTrainer(uml_gpt, get_dataloaders(tokenized_dataset, args.batch_size), args)
    print("Done!")

    print("Training...")
    trainer.train(args.num_epochs)
    trainer.save_model(f'{args.trainer}_uml_gpt_{args.num_epochs}.pt')


def get_classification_model(model_name, num_labels, tokenizer):
    if 'bert' in model_name:
        model = AutoModelForSequenceClassification.from_pretrained(pretrained_model_name_or_path=model_name, num_labels=num_labels, ignore_mismatched_sizes=True)
        model.resize_token_embeddings(len(tokenizer))
    else:
        model_config = GPT2Config.from_pretrained(pretrained_model_name_or_path=model_name, num_labels=num_labels)
        tokenizer.padding_side = "left"
        model = GPT2ForSequenceClassification.from_pretrained(pretrained_model_name_or_path=model_name, config=model_config)
        model.resize_token_embeddings(len(tokenizer)) 
        model.config.pad_token_id = model.config.eos_token_id
        
    return model


def train_hf_for_classification(tokenizer, dataset, args):
    model_name = args.model_name
    batch_size = args.batch_size
    train, test, unseen = dataset['train'], dataset['test'], dataset['unseen']
    # Show the training loss with every epoch
    logging_steps = len(train) // batch_size
    print(f"Using model...{model_name}")
    model = get_classification_model(model_name, len(dataset.num_labels), tokenizer)
    model.resize_token_embeddings(len(tokenizer))
    print("Finetuning model...")
    training_args = TrainingArguments(
        output_dir=args.out_dir,
        overwrite_output_dir=True,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        learning_rate=2e-5,
        weight_decay=0.01,
        warmup_steps=args.warmup_steps,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        fp16=True,
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

    trainer.train()
    print("Evaluating on test set...")
    print(trainer.evaluate(test))
    print("Evaluating on unseen set...")
    print(trainer.evaluate(unseen))

    trainer.save_model(os.path.join(args.out_dir, f'uml_{model_name}'))