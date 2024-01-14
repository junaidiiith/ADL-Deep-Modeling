from constants import *
import numpy as np
from transformers import AutoTokenizer
from vocab_tokenizer import VocabTokenizer
from datasets import EncodingsDataset
from models import UMLGPTClassifier
from stqdm import stqdm
import transformers
import pandas as pd
import streamlit as st

from uml_data_generation import get_gpt2_dataset
from uml_data_generation import get_dataloaders
from training_utils import get_pretrained_lm_tokenizer, get_word_tokenizer_tokenizer
from uml_data_generation import get_generative_uml_dataset

import pickle
from models import UMLGPT
from transformers import \
    GPT2Config, \
    GPT2ForSequenceClassification, \
    AutoModelForSequenceClassification, \
    AutoModelForCausalLM

from transformers import TrainingArguments, Trainer, DataCollatorForLanguageModeling
from transformers.integrations import NeptuneCallback
from trainers import UMLGPTTrainer
from metrics import compute_metrics


def suppress_neptune(trainer):
    for cb in trainer.callback_handler.callbacks:
        if isinstance(cb, NeptuneCallback):
            trainer.callback_handler.remove_callback(cb)


def compute_loss(pos_score, neg_score):
    scores = torch.cat([pos_score, neg_score]).to(DEVICE)
    labels = torch.cat([torch.ones(pos_score.shape[0]), torch.zeros(neg_score.shape[0])]).to(DEVICE)
    return torch.nn.BCEWithLogitsLoss()(scores.float(), labels.float())


def get_tokenization(tokenizer, data):
    """
    ``get_tokenization`` function returns the tokenization for the given tokenizer and data
    """
    if isinstance(tokenizer, VocabTokenizer):
        tokenized_data = tokenizer.batch_encode(
            data, return_tensors='pt', max_length='percentile')
    else:
        tokenized_data = tokenizer(
            data, return_tensors='pt', padding=True)
    return tokenized_data


def get_embedding(model, encodings, pooling='last'):
    """
    ``get_embedding`` function returns the embeddings for the given model and encodings
    pooling: last, mean, max, min, sum, cls
    pooling is used to pool the embeddings of the tokens in the sequence
    e.g., if pooling is last, the last token embedding is used as the embedding for the sequence
    if pooling is mean, the mean of the token embeddings is used as the embedding for the sequence
    """
    encoding_dataset = EncodingsDataset(encodings)
    encoding_dataloader = torch.utils.data.DataLoader(encoding_dataset, batch_size=128, shuffle=False)
    model.eval()

    with torch.no_grad():
        embeddings = list()
        for batch in encoding_dataloader:

            if isinstance(model, UMLGPT) or isinstance(model, UMLGPTClassifier):
                outputs = model.get_embedding(batch['input_ids'].to(DEVICE), batch['attention_mask'].to(DEVICE))
            else:
                encodings = {k: v.to(DEVICE) for k, v in batch.items()}
                outputs = model(**encodings)[0]

            outputs = outputs.cpu().detach()
            if pooling == 'last':
                outputs = outputs[:, -1, :]
            elif pooling == 'mean':
                outputs = torch.mean(outputs, dim=1)
            elif pooling == 'max':
                outputs = torch.max(outputs, dim=1)[0]
            elif pooling == 'min':
                outputs = torch.min(outputs, dim=1)[0]
            elif pooling == 'sum':
                outputs = torch.sum(outputs, dim=1)
            elif pooling == 'cls':
                outputs = outputs[:, 0, :]
            else:
                raise ValueError(f"Pooling {pooling} not supported")
            embeddings.append(outputs)
        
        embeddings = torch.cat(embeddings, dim=0)
        
    return embeddings



def get_encoding_size(data, tokenizer):
    """
    ``get_encoding_size`` function returns the encoding size for the given data and tokenizer
        i.e., given a list of strings, it returns the 99.5th percentile of the lengths of the tokenized strings
    99.5th percentile is used to avoid the out of memory error while training the model
    """

    tokens = tokenizer(data)
    lengths = [len(i) for i in tokens['input_ids']]
    size = int(np.percentile(lengths, 99.5))
    # print("Encoding size: ", size)
    return size


def get_pretrained_lm_tokenizer(model_name, special_tokens=SPECIAL_TOKENS):
    """
        ``get_pretrained_lm_tokenizer`` function returns the tokenizer for the given hugging face language model
    """

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.add_special_tokens({'additional_special_tokens': special_tokens})
    if tokenizer.pad_token is None:
        tokenizer.pad_token = "<pad>"

    print("Vocab size: ", len(tokenizer))
    return tokenizer


def get_word_tokenizer_tokenizer(data, lower=True, special_tokens=SPECIAL_TOKENS):
    """
    ``get_word_tokenizer_tokenizer`` function constructs a custom Vocabulary tokenizer for the given data
    """

    tokenizer = VocabTokenizer(data, lower=lower, special_tokens=special_tokens)
    print("Vocab size: ", len(tokenizer))
    return tokenizer


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
    if args.tokenizer_file is not None:
        tokenizer = pickle.load(open(args.tokenizer_file, 'rb'))
    
    elif args.tokenizer == WORD_TOKENIZER:
        tokenizer = get_tokenizer(WORD_TOKENIZER, dataset)
        tokenizer.save_pretrained(args.models_dir)
        print(f"Saved tokenizer at {args.models_dir}")
    
    else:
        tokenizer = get_tokenizer(args.tokenizer)


    print("Tokenize dataset...")
    tokenized_dataset = get_generative_uml_dataset(dataset, tokenizer)
    print("Done!")

    uml_gpt = get_uml_gpt(len(tokenizer), args)

    print("Model initialized! with parameters:")
    print("Batch size: ", args.batch_size)
    dataloaders = get_dataloaders(tokenized_dataset, args.batch_size)

    print("Creating dataloaders and trainer...")
    trainer = UMLGPTTrainer(uml_gpt, dataloaders, args)
    print("Done!")
    if args.phase == TRAINING_PHASE:
        print("Training...")
        trainer.train(args.num_epochs)
        trainer.save_model(f'final_model.pt')
    else:
        print("Evaluating: ", len(dataloaders[TEST_LABEL].dataset))
        trainer.evaluate()


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
        model = GPT2ForSequenceClassification.from_pretrained(
            pretrained_model_name_or_path=model_name, 
            config=model_config, 
            ignore_mismatched_sizes=True
        )
        model.resize_token_embeddings(len(tokenizer)) 
        model.config.pad_token_id = model.config.eos_token_id
    else:
        model = AutoModelForSequenceClassification.from_pretrained(
            pretrained_model_name_or_path=model_name, num_labels=num_labels, ignore_mismatched_sizes=True)
        model.resize_token_embeddings(len(tokenizer))
        
    return model


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
        fp16=True,
        greater_is_better=False
    )

    trainer = Trainer(
        model=model,                        
        args=training_args,                 
        train_dataset=dataset[TRAIN_LABEL] if args.phase == TRAINING_PHASE else None,
        eval_dataset=dataset[TEST_LABEL],          # evaluation dataset
        data_collator=data_collator,
        tokenizer=tokenizer,
    )

    suppress_neptune(trainer)

    st_results = st.empty()
    results = list()
    if args.phase == TRAINING_PHASE:
        best_loss = float('inf')
        # for epoch in tqdm(range(args.num_epochs), desc="Training Epoch: "):
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
                # with st.spinner("Saving best model..."):
                #     trainer.save_model(args.models_dir)
            
            
            test_results = {f"test_{k}": v for k, v in test_results.items()}
            unseen_results = {f"unseen_{k}": v for k, v in unseen_results.items()}
            results.append({**{'Epoch': epoch}, **test_results, **unseen_results})

            with st_results.container():
                st.markdown(f"### {epoch+1} Results")
                st.dataframe(pd.DataFrame(results), hide_index=True)
        
    else:
        test_results = {'eval_loss': 0, 'eval_accuracy': 0}
        # test_results = trainer.evaluate(dataset[TEST_LABEL])
        results.append(test_results)

    # with st_results.container():
    #     st.markdown(f"## Pretraining Results")
    #     st.dataframe(pd.DataFrame(results), hide_index=True)


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

    best_loss = float('inf')
    results = list()
    st_results = st.empty()
    for epoch in stqdm(range(args.num_epochs), desc="Training Epoch: "):
    # for epoch in tqdm(range(args.num_epochs), desc="Training Epoch: "):
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
            # with st.spinner("Saving best model ..."):
            #     trainer.save_model(args.models_dir)
            
        
        test_results = {f"test_{k}": v for k, v in test_results.items()}
        unseen_results = {f"unseen_{k}": v for k, v in unseen_results.items()}
        results.append({**{'Epoch': epoch}, **test_results, **unseen_results})

        with st_results.container():
            st.markdown(f"### Epoch {epoch+1} Results")
            st.dataframe(pd.DataFrame(results), hide_index=True)
    
    with st_results.container():
        st.markdown(f"### HF Classification {epoch+1} Results")
        st.dataframe(pd.DataFrame(results), hide_index=True)
