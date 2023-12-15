import os
import pickle
import random
from tqdm.auto import tqdm


def get_node_triples(graphs_file):
    graph_file_name = os.path.basename(graphs_file)
    node_triples_file = os.path.join(os.path.dirname(graphs_file), f'{graph_file_name}_node_triples.pkl')
    if os.path.exists(node_triples_file):
        node_triples = pickle.load(open(node_triples_file, 'rb'))
        return node_triples

    graphs = pickle.load(open(graphs_file, 'rb'))
    node_triples = set()

    for g in tqdm(graphs):
        for n in g.nodes():
            super_type_nodes = [edge[1] for edge in g.edges(n) if g.edges[edge]['type'] == 'generalization' and len(edge[1].strip())]

            if 'NamedElement' in super_type_nodes:
                super_type_nodes.remove('NamedElement')

            reference_nodes = [edge[1] for edge in g.edges(n) if g.edges[edge]['type'] != 'generalization' and len(edge[1].strip())]
            if not len(reference_nodes):
                continue

            selected_super_types = random.sample(super_type_nodes, min(5, len(super_type_nodes)))

            node_references = [edge[1] for edge in g.edges(n) if g.edges[edge]['type'] != 'generalization']
            for node_reference in node_references:
                edge_name = g.edges[n, node_reference]['name'] if 'name' in g.edges[n, node_reference] else ''
                node_triples.add((n, edge_name, node_reference, ", ".join(selected_super_types)))

    node_triples = list(node_triples)
    print(len(node_triples))

    pickle.dump(node_triples, open(node_triples_file, 'wb'))

    return node_triples
