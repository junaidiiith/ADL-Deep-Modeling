import pickle
import os


def set_uploaded_file_path(args, graph_file):
    args.graphs_file = os.path.join(args.uploaded_data_dir, graph_file.name)
    
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
