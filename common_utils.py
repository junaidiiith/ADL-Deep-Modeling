from ast import arg
import os
import json
import random
from streamlit.runtime.uploaded_file_manager import UploadedFile
import numpy as np
import re
import torch
from constants import *


clean_text = lambda x: re.sub(r'[^a-zA-Z0-9\s]', '', x).strip()

def set_seed(seed):
    """
        This method sets the seed for random, numpy, and torch libraries
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)


def get_attr_name(uploaded_file):
    """
        This method returns the uploaded file from the streamlit file uploader
    """
    if isinstance(uploaded_file, UploadedFile):
        return uploaded_file.name
    return uploaded_file


def create_run_command_line(args):
    arguments = ['python', task2file_map[args.stage]]
    arguments += [f"--{k}={getattr(args, k)}" for k in vars(args) if getattr(args, k) is not None]
    return " ".join(arguments)


def create_run_config(args):
    """
        This method creates a run config for the given arguments
    """
    set_seed(args.seed)
    config = {k: get_attr_name(getattr(args, k)) for k in vars(args)}
    config[RUN_COMMAND] = create_run_command_line(args)

    print(config[RUN_COMMAND])

    file_name = ""

    if args.stage == PRETRAINING:
        ## If stage is pretraining then there has to be a pretraining gpt model
        if config[GPT_MODEL] in [UMLGPTMODEL]:
            file_name += f"{config[GPT_MODEL]}_tok={config['tokenizer']}"
        else:
            file_name += f"{config[GPT_MODEL]}"


    elif args.stage == UML_CLASSIFICATION:

        if args.classification_model in [UMLGPTMODEL]:

            if config[FROM_PRETRAINED] is not None and UML_CLASSIFICATION not in config[FROM_PRETRAINED]:
                file_name += f"fp_{config[FROM_PRETRAINED].replace(os.sep, '_').replace(BEST_MODEL_LABEL, '')}"
            else:
                file_name += f"{config[CLASSIFICATION_MODEL]}" + f"_tok={config['tokenizer']}"
        else:
            model_name = os.path.basename(config[FROM_PRETRAINED])
            file_name += f"fp_{model_name}" if UML_CLASSIFICATION not in model_name else "_inf_"
        
        file_name += f"_{config[CLASSIFICATION_TYPE]}"

        
    elif args.stage == LINK_PREDICTION:
        file_name += f"{config[EMBEDDING_MODEL]}"
        
    
    elif args.stage == ONTOML_CLS:
        if args.phase == TRAINING_PHASE:
            if UML_CLASSIFICATION not in config[FROM_PRETRAINED]:
                file_name += f"fp_{config[FROM_PRETRAINED]}"
            else:
                file_name += f"fp_{config[FROM_PRETRAINED].split(os.sep)[-1]}"
            
            file_name += f"_distance_{args.distance}"
            file_name += f"_el_{args.exclude_limit}"
        else:
            file_name += f"{config[FROM_PRETRAINED].split(os.sep)[-1]}"
        
            

    
    os.makedirs(os.path.join(args.log_dir, file_name), exist_ok=True)
    args.log_dir = os.path.join(args.log_dir, file_name)
    
    os.makedirs(os.path.join(args.models_dir, file_name), exist_ok=True)
    args.models_dir = os.path.join(args.models_dir, file_name)

    args.config_file_name = file_name

    # print(config)
    # print(args.models_dir)

    json.dump(config, open(os.path.join(args.models_dir, f'{TRAINING_CONFIG_JSON}'), 'w'), indent=4)

    return config
