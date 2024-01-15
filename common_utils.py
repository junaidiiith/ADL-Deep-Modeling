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
    
    file_name = f"{args.stage}_"
    if args.stage == PRETRAINING:
        if config[GPT_MODEL] in [UMLGPTMODEL]:
            file_name += f"{config[GPT_MODEL]}_tok={config['tokenizer']}"
        else:
            file_name += f"{config[GPT_MODEL]}"

    elif args.stage == UML_CLASSIFICATION:
        if args.classification_model not in [UMLGPTMODEL]:
            file_name += f"fp_{config[FROM_PRETRAINED].split(os.sep)[-2]}"
        else:
            if config[FROM_PRETRAINED] is not None:
                file_name += f"fp_{config[FROM_PRETRAINED].split(os.sep)[-2]}"
            else:
                file_name += f"{config[CLASSIFICATION_MODEL]}"

            file_name += f"_tok={config['tokenizer']}"
        
        file_name += f"_{config[CLASSIFICATION_TYPE]}"
        

    elif args.stage == LINK_PREDICTION:
        file_name += f"{config[EMBEDDING_MODEL].split(os.sep)[-2]}_tok={config['tokenizer']}"
    
    elif args.stage == ONTOML_CLS:
        if config[FROM_PRETRAINED] is not None and args.phase == INFERENCE_PHASE:
            file_name += f"_fp_{config[FROM_PRETRAINED].split(os.sep)[-2]}"
        file_name += f"_distance={args.distance}"
        file_name += f"_distance={args.exclude_limit}"

    
    os.makedirs(os.path.join(args.log_dir, file_name), exist_ok=True)
    args.log_dir = os.path.join(args.log_dir, file_name)
    
    os.makedirs(os.path.join(args.models_dir, file_name), exist_ok=True)
    args.models_dir = os.path.join(args.models_dir, file_name)

    args.config_file_name = file_name

    # print(config)
    # print(args.models_dir)

    json.dump(config, open(os.path.join(args.models_dir, f'config.json'), 'w'), indent=4)

    return config
