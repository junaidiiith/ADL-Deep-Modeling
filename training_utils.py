from constants import *
import pandas as pd
import streamlit as st

from uml_data_generation import get_gpt2_dataset
from uml_data_generation import get_dataloaders
from tokenization import get_pretrained_lm_tokenizer, get_word_tokenizer_tokenizer
from uml_data_generation import get_generative_uml_dataset

import pickle
from models import UMLGPT
from transformers import \
    GPT2Config, \
    GPT2ForSequenceClassification, \
    AutoModelForSequenceClassification, \
    AutoModelForCausalLM

from trainers.umlgpt import UMLGPTTrainer
from metrics import get_recommendation_metrics
from trainers.causal_lm import CausalLMTrainer
from trainers.hf_classifier import ClassificationTrainer



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
        eval_results = trainer.evaluate()
        print(eval_results)
        st.dataframe(pd.DataFrame([eval_results]), hide_index=True)

    

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

    trainer = CausalLMTrainer(model, tokenizer, dataset, args)
    if args.phase == TRAINING_PHASE:
        trainer.train(args.num_epochs)
        trainer.save_model()
    else:
        results = trainer.evaluate()
        st.dataframe([results], hide_index=True)



        
    