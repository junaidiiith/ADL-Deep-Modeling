import pickle
import os
import shutil
from zipfile import ZipFile



def unzip_models(input_file, extension, args):
    """
        This method unzips the graphs file and sets the path to the graphs file
    """
    f_name = input_file.name.split('.')[0]
    if input_file.name.endswith('.zip'):
        
        with ZipFile(input_file, 'r') as zip:
            zip.extractall(args.data_dir)
        args.graphs_file = os.path.join(args.data_dir, f_name)

    elif input_file.endswith(f'{extension}'):
        dir_path = os.path.join(args.data_dir, f_name)
        os.makedirs(dir_path, exist_ok=True)
        shutil.copy(input_file, dir_path, input_file.name)
        args.graphs_file = dir_path
        

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
    # print(models_dir, task_type, model_name)
    plms = [
        f for f in os.listdir(models_dir) \
        if os.path.isdir(os.path.join(models_dir, f)) and \
        (f.startswith(f'{task_type}_{model_name}') or f.startswith(f'{task_type}_fp_{model_name}'))
    ]
    return plms
