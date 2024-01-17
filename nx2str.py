import dgl
import torch
from collections import Counter
import os
import pickle
import random
from sklearn.model_selection import train_test_split
from tqdm.auto import tqdm
from common_utils import clean_text
from model2nx import get_graphs_from_directory

edge_name = lambda g, edge: g.edges[edge]['name'] if 'name' in g.edges[edge] else ''


def masked_graph(graph, mask_perc=0.2):
    """
    Mask graph by setting 20% of the edges to masked
    """

    for edge in graph.edges():
        graph.edges[edge]['masked'] = False

    edges = list(graph.edges())
    num_edges_to_mask = int(len(edges) * mask_perc)
    edges_to_mask = random.sample(edges, num_edges_to_mask)
    
    for edge in edges_to_mask:
        graph.edges[edge]['masked'] = True
    
    assert all(['type' in graph.edges[edge] for edge in graph.edges()])
    

def add_node_connections_str(g):

    """
    Add references and super types to the nodes in the graph
    This information is later used to generate the node triples
    For e.g., if a triple would be (node, reference, super_type)
    """
    assert all(['type' in g.edges[edge] for edge in g.edges()])
    edges_to_remove = [edge for edge in g.edges() if g.edges[edge]['masked']]
    edge_dicts = {edge: g.edges[edge] for edge in edges_to_remove}
    g.remove_edges_from(edges_to_remove)
    

    for n in g.nodes():
        super_type_nodes = [edge[1] for edge in g.edges(n) if g.edges[edge]['type'] == 'generalization' and len(edge[1].strip())]

        if 'NamedElement' in super_type_nodes:
            super_type_nodes.remove('NamedElement')

        reference_edges = [edge for edge in g.edges(n) if g.edges[edge]['type'] != 'generalization' and len(edge[1].strip())]

        if len(super_type_nodes) == 0 and len(reference_edges) == 0:
            continue

        references = ", ".join([f"{edge_name(g, edge)} {edge[1]}" for edge in reference_edges])
        super_types_str = ", ".join(super_type_nodes)

        g.nodes[n]['references'] = references
        g.nodes[n]['super_types'] = super_types_str
    
    g.add_edges_from(edges_to_remove)
    for edge in edges_to_remove:
        g.edges[edge]['masked'] = True
        for k, v in edge_dicts[edge].items():
            g.edges[edge][k] = v
        

def mask_graphs(graphs):
    """
        Mask a list of graphs
    """
    for graph in graphs:
        assert all(['type' in graph.edges[edge] for edge in graph.edges()])

    for graph in tqdm(graphs, desc='Masking graphs'):
        masked_graph(graph)
    

def add_node_connections_str_to_graphs(graphs):
    """
        Mask a list of graphs
        Add node connections to a list of graphs
    """
    for graph in graphs:
        assert all(['type' in graph.edges[edge] for edge in graph.edges()])
        

    mask_graphs(graphs)
    for g in tqdm(graphs, desc='Adding node strings to graphs'):
        add_node_connections_str(g)
    

def get_node_triples_from_graph(g):
    """
        Get node triples from a single graph
    """
    triples = list()

    for n in g.nodes():
        if 'super_types' not in g.nodes[n]:
            continue
        super_types = g.nodes[n]['super_types']
        references = g.nodes[n]['references']

        triples.append((n, references, super_types))

    return triples

def filter_graphs(graphs):
    filtered_graphs = list()
    for g in graphs:
        if any(['masked' in g.edges[edge] for edge in g.edges()]):
            filtered_graphs.append(g)
            continue
    
    graphs.clear()
    graphs.extend(filtered_graphs)
    return graphs


def get_node_triples_from_graphs(graphs):
    """
        Get node triples from a list of graphs
    """
    remove_duplicates = lambda l: list({"$$".join(t): t for t in l}.values())
    triples = list()

    for g in tqdm(graphs, desc='Getting node triples'):
        graph_triples = get_node_triples_from_graph(g)

        for triple in graph_triples:
            triples.append(triple)

    return remove_duplicates(triples)


def get_graphs_from_dir(graphs_file):
    graphs_file_dir = os.path.dirname(graphs_file)
    graphs_file_name = os.path.basename(graphs_file).split('.')[0]
    graphs = pickle.load(open(graphs_file, 'rb'))
    node_triples_file = os.path.join(graphs_file_dir, f'{graphs_file_name}_node_triples.pkl')
    return graphs, node_triples_file


def get_graph_data(graphs_file, seed=42):
    """
        Takes a graphs dir path containing a list of graphs
        Split the graphs into train and test
        Adds node connections to the graphs i.e., references and super types
        Get node triples from the graphs i.e., (node, references, super types)
        Encode the entities and super types i.e., map them to integers

        For each node, it can have multiple super types therefore we select the super type
        with the highest frequency in the dataset
        
        Args:
            graphs_file: pickle file containing a list of graphs
            seed: random seed for splitting the graphs
            
        Returns:
            train_graphs: list of train graphs
            test_graphs: list of test graphs
            train_triples: list of train triples
            test_triples: list of test triples
            entities_encoder: dictionary mapping entities to integers
            super_types_encoder: dictionary mapping super types to integers

    """
    graphs = get_graphs_from_directory(graphs_file)
    if isinstance(graphs[0], tuple):
        graphs = [g for _, g in graphs]

    for graph in graphs:
        assert all(['type' in graph.edges[edge] for edge in graph.edges()])

    dir_name = os.path.dirname(graphs_file)
    node_triples_file = os.path.join(dir_name, os.path.basename(graphs_file) + '_node_triples.pkl')

    if os.path.exists(node_triples_file):
        data = pickle.load(open(node_triples_file, 'rb'))
        return data

    if len(graphs) > 1:    
        train_graphs, test_graphs = train_test_split(graphs, test_size=0.05, random_state=seed)
    else:
        train_graphs, test_graphs = [graph.copy()], [graph.copy()]


    for graph in train_graphs:
        assert all(['type' in graph.edges[edge] for edge in graph.edges()])

    for graph in test_graphs:
        assert all(['type' in graph.edges[edge] for edge in graph.edges()])
        
    
    add_node_connections_str_to_graphs(train_graphs)
    add_node_connections_str_to_graphs(test_graphs)

    train_graphs = filter_graphs(train_graphs)
    test_graphs = filter_graphs(test_graphs)

    print("Total train graphs:", len(train_graphs))
    print("Total test graphs:", len(test_graphs))

    train_triples = get_node_triples_from_graphs(train_graphs)
    test_triples = get_node_triples_from_graphs(test_graphs)

    print("Sample Train triples", train_triples[:2])
    print("Sample Test triples", test_triples[:2])

    all_entities = [clean_text(t[0]) for t in train_triples] + [clean_text(t[0]) for t in test_triples]
    all_super_types = [i.split() for i in [clean_text(t[-1]) for t in train_triples] + [clean_text(t[-1]) for t in test_triples]]
    
    all_entities_count = Counter(all_entities)
    all_super_types_count = Counter([j for i in all_super_types for j in i])

    all_selected_super_types = [
        max(super_types, key=lambda x: all_super_types_count[x])\
            if len(super_types) else ''  for super_types in all_super_types]
    
    selected_super_types_count = Counter(all_selected_super_types)
    selected_super_types_count.pop('', None)

    assert all([a in all_super_types_count for a in selected_super_types_count]), "Some selected super types are not in the super types count"

    print("Total entities:", len(all_entities_count))
    print("Total super types:", len(selected_super_types_count))

    entities_encoder = {v: i for i, v in enumerate(all_entities_count.keys())}
    super_types_encoder = {v: i for i, v in enumerate(selected_super_types_count.keys())}
    assert len(train_triples) + len(test_triples) == len(all_selected_super_types), "Some super types are missing"
    train_triples = [(t[0], t[1], super_type) for t, super_type in zip(train_triples, all_selected_super_types[:len(train_triples)])]
    test_triples = [(t[0], t[1], super_type) for t, super_type in zip(test_triples, all_selected_super_types[len(train_triples):])]

    print("Sample Train triples", train_triples[:2])
    print("Sample Test triples", test_triples[:2])
    print("Total train triples:", len(train_triples))
    print("Total test triples:", len(test_triples))
    
    data = {
        'train_graphs': train_graphs,
        'test_graphs': test_graphs,
        'train_triples': train_triples,
        'test_triples': test_triples,
        'entities_encoder': entities_encoder,
        'super_types_encoder': super_types_encoder,
    }

    pickle.dump(data, open(node_triples_file, 'wb'))
    
    
    return data

# if __name__ == "__main__":
#     random.seed(42)
#     data = get_graph_data('datasets/test_data_graphs.pkl')
#     triples = json.dump({'train_data': data['train_triples'], 'test_data': data['test_triples']}, open(f'triples.json', 'w'), indent=4)