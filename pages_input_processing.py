import pickle
import os
import shutil
from zipfile import ZipFile
import streamlit as st

from constants import BEST_MODEL_LABEL, SAFE_TENSORS_LABEL


def unzip_models(input_file, extension, args):
    """
        This method unzips the graphs file and sets the path to the graphs file
    """
    print("Input File", input_file.name)
    
    f_name = input_file.name.split('.')[0]
    dir_path = os.path.join(args.data_dir, f_name)
    
    if input_file.name.endswith('.zip') and not os.path.exists(dir_path):
        with st.spinner('Unzipping graphs file'):
            with ZipFile(input_file, 'r') as zip:
                zip.extractall(args.data_dir)
        

    elif input_file.name.endswith(f'{extension}'):
        os.makedirs(dir_path, exist_ok=True)
        with open(os.path.join(dir_path, input_file.name), 'wb') as f:
            f.write(input_file.getvalue())
        
    
    args.graphs_file = dir_path
    print("graph file", args.graphs_file)
        

def set_uploaded_file_path(args, graph_file):
    args.graphs_file = os.path.join(args.data_dir, graph_file.name)
    
    if not os.path.exists(args.graphs_file):
        graphs = pickle.loads(graph_file.getvalue())
        with open(args.graphs_file, 'wb') as f:
            pickle.dump(graphs, f)
    

def get_plms(models_dir, task_type, model_name):
    """
        This method returns the list of pre-trained language models
        for the given task type and model name available in the given models directory
    """
    print([f for f in os.listdir(models_dir)])
    print(models_dir, task_type, model_name)
    plms = [
        f for f in os.listdir(models_dir) \
        if os.path.isdir(os.path.join(models_dir, f)) and \
        model_name in f and not f'tok={model_name}' in f and \
        (
            BEST_MODEL_LABEL in [x for x in os.listdir(os.path.join(models_dir, f))] or \
            SAFE_TENSORS_LABEL in [x for x in os.listdir(os.path.join(models_dir, f))]
        )
    ]
    
    return plms
