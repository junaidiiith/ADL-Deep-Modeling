import pickle
from constants import UPLOADED_DATA_DIR
import os


def set_uploaded_file_path(args, graph_file):
    args.graphs_file = os.path.join(UPLOADED_DATA_DIR, graph_file.name)
    
    if not os.path.exists(args.graphs_file):
        graphs = pickle.loads(graph_file.getvalue())
        with open(UPLOADED_DATA_DIR, graph_file.name, 'wb') as f:
            pickle.dump(graphs, f)
    
    