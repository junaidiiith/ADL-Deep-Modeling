from collections import Counter
import json
import os
import pickle
import random
from sklearn.model_selection import train_test_split
from tqdm.auto import tqdm
from utils import clean_text


edge_name = lambda g, edge: g.edges[edge]['name'] if 'name' in g.edges[edge] else ''

def get_masked_graph(graph, seed=42):
    """
    Mask graph by removing 20% of the edges
    """

    masked_graph = graph.copy()

    random.seed(seed)
    edges = list(masked_graph.edges())
    random.shuffle(edges)

    num_edges_to_remove = int(len(edges) * 0.2)

    for edge in edges[:num_edges_to_remove]:
        masked_graph.remove_edge(edge[0], edge[1])

    return masked_graph


def add_node_connections_str(g, max_super_types=5):
    for n in g.nodes():
        super_type_nodes = [edge[1] for edge in g.edges(n) if g.edges[edge]['type'] == 'generalization' and len(edge[1].strip())]

        if 'NamedElement' in super_type_nodes:
            super_type_nodes.remove('NamedElement')

        reference_edges = [edge for edge in g.edges(n) if g.edges[edge]['type'] != 'generalization' and len(edge[1].strip())]
        selected_super_types = random.sample(super_type_nodes, min(max_super_types, len(super_type_nodes)))

        if len(selected_super_types) == 0 and len(reference_edges) == 0:
            continue

        references = ", ".join([f"{edge_name(g, edge)} {edge[1]}" for edge in reference_edges])
        super_types_str = ", ".join(selected_super_types)

        g.nodes[n]['references'] = references
        g.nodes[n]['super_types'] = super_types_str


def get_graphs_with_node_str(graphs, max_super_types=5, seed=42):
    graph_triples = list()

    for ecore_graph in tqdm(graphs, desc='Adding node strings to graphs'):
        g = get_masked_graph(ecore_graph, seed)
        add_node_connections_str(g, max_super_types)
        graph_triples.append(g)
    
    return graph_triples


def get_node_triples_from_graph(g):
    triples = list()

    for n in g.nodes():
        if 'super_types' not in g.nodes[n]:
            continue
        super_types = g.nodes[n]['super_types']
        references = g.nodes[n]['references']

        triples.append((n, references, super_types))

    return triples


def get_node_triples_from_graphs(graphs):
    triples = set()

    for g in tqdm(graphs, desc='Getting node triples'):
        graph_triples = get_node_triples_from_graph(g)
        for triple in graph_triples:
            triples.add(triple)

    return list(triples)



def get_graph_data(graphs_file, max_super_types=5, seed=42):
    graph_file_name = os.path.basename(graphs_file).split('.')[0]

    node_triples_file = os.path.join(os.path.dirname(graphs_file), f'{graph_file_name}_node_triples.pkl')
    if os.path.exists(node_triples_file):
        data = pickle.load(open(node_triples_file, 'rb'))
        return data

    graphs = pickle.load(open(graphs_file, 'rb'))

    train_graphs, test_graphs = train_test_split(graphs, test_size=0.05, random_state=seed)

    train_graphs_masked = get_graphs_with_node_str(train_graphs, max_super_types, seed)
    test_graphs_masked = get_graphs_with_node_str(test_graphs, max_super_types, seed)

    train_triples = get_node_triples_from_graphs(train_graphs_masked)
    test_triples = get_node_triples_from_graphs(test_graphs_masked)

    all_entities = [clean_text(t[0]) for t in train_triples] + [clean_text(t[0]) for t in test_triples]
    all_super_types = [clean_text(st) for t in train_triples for st in t[2].split(', ')]\
          + [clean_text(st) for t in test_triples for st in t[2].split(', ')]

    all_entities_count = Counter(all_entities)
    all_super_types_count = Counter(all_super_types)

    print("Total entities:", len(all_entities_count))
    print("Total super types:", len(all_super_types_count))

    entities_encoder = {v: i for i, v in enumerate(all_entities_count.keys())}
    super_types_encoder = {v: i for i, v in enumerate(all_super_types_count.keys())}

    # for x in train_triples + test_triples:
    #     assert f"{clean_text(x[2])}".split() == x[2].replace(",", "").split(), f"{f'{clean_text(x[2])}'.split()} != {x[2].replace(' ', '').split()}"

    data = {
        'train_graphs': train_graphs,
        'test_graphs': test_graphs,
        'train_graphs_masked': train_graphs_masked,
        'test_graphs_masked': test_graphs_masked,
        'train_triples': train_triples,
        'test_triples': test_triples,
        'entities_encoder': entities_encoder,
        'super_types_encoder': super_types_encoder,
    }

    pickle.dump(data, open(node_triples_file, 'wb'))
    return data



if __name__ == "__main__":
    data = get_graph_data('datasets/ecore_graph_pickles/combined_graphs_clean.pkl')

    print("Training triples:", len(data['train_triples']))
    print("Test triples:", len(data['test_triples']))

    triples = json.dump({'train_data': data['train_triples'], 'test_data': data['test_triples']}, open(f'triples.json', 'w'), indent=4)