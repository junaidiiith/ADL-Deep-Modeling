import pickle
import os
from zipfile import ZipFile
from constants import UPLOADED_DATA_DIR
from model2nx import get_graph_from_files
from model2nx import clean_graph_set


def unzip_graphs_file(zip_file, extension, data_dir='.'):
    """
        This method unzips the graphs file and sets the path to the graphs file
    """
    all_files = list()
    with ZipFile(zip_file, 'r') as zip:
        zip.extractall(data_dir)
        for root, _, files in os.walk("./"):
            for file in files:
                if file.endswith(f".{extension}"):
                    all_files.append(os.path.join(root, file))
    
    return all_files


def unzip_ecore_models(zip_file, args):
    """
        This method unzips the graphs file and sets the path to the graphs file
    """
    all_files = unzip_graphs_file(zip_file, 'ecore', args.data_dir)
    graphs = get_graph_from_files(all_files)
    cleaned_graphs = clean_graph_set(graphs)

    args.graphs_file = os.path.join(args.data_dir, zip_file.name)

    if not os.path.exists(args.graphs_file):
        with open(args.graphs_file, 'wb') as f:
            pickle.dump(cleaned_graphs, f)


def unzip_ontouml_models(zip_file, args):
    """
        This method unzips the graphs file and sets the path to the graphs file
    """
    unzip_graphs_file(zip_file, 'json', args.data_dir)
    args.data_dir = os.path.join(args.data_dir,  zip_file.name.split('.')[0])
    

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
    plms = [
        f for f in os.listdir(models_dir) \
        if os.path.isdir(os.path.join(models_dir, f)) and \
        f.startswith(f'{task_type}_{model_name}')
    ]
    return plms
