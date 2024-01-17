import fnmatch
import pickle
from pyecore.resources import ResourceSet, URI
import networkx as nx

import os
import networkx as nx
from stqdm import stqdm



def find_files_with_extension(root_dir, extension):
    matching_files = []

    # Recursively search for files with the specified extension
    for root, _, files in os.walk(root_dir):
        for filename in fnmatch.filter(files, f'*.{extension}'):
            matching_files.append(os.path.join(root, filename))

    return matching_files


def get_attributes(classifier):
    all_feats = set((feat.name, feat.eType.name) for feat in classifier.eAllStructuralFeatures() if type(feat).__name__ == 'EAttribute')
    return list(all_feats)

def get_model_root(file_name):
    rset = ResourceSet()
    resource = rset.get_resource(URI(file_name))
    mm_root = resource.contents[0]
    return mm_root


def get_ecore_data(file_name):
    rset = ResourceSet()
    resource = rset.get_resource(URI(file_name))
    mm_root = resource.contents[0]
    references = list()
    for classifier in mm_root.eClassifiers:
        # print(classifier.name, get_features(classifier))
        if type(classifier).__name__ == 'EClass':
            references.append((classifier.name, get_attributes(classifier)))
    super_types = list()
    for classifier in mm_root.eClassifiers:
        if type(classifier).__name__ == 'EClass':
            for supertype in classifier.eAllSuperTypes():
                super_types.append((classifier.name, supertype.name))
    return references, super_types


def create_nx_from_ecore(file_name):
    print("Creating graph for: ", file_name)
    count = 0
    try:
        model_root = get_model_root(file_name)
    except Exception as e:
        return None
    if type(model_root).__name__ != 'EPackage':
        return None
    nxg = nx.DiGraph()
    for classifier in model_root.eClassifiers:
        if type(classifier).__name__ == 'EClass':
            if not nxg.has_node(classifier.name):
                nxg.add_node(classifier.name, name=classifier.name, type='class')

            classifier_attrs = set(feat.name for feat in classifier.eAllStructuralFeatures() if type(feat).__name__ == 'EAttribute')
            nxg.nodes[classifier.name]['attrs'] = list(classifier_attrs)
    
    for classifier in model_root.eClassifiers:
        if type(classifier).__name__ == 'EClass':
            for supertype in classifier.eAllSuperTypes():
                if not nxg.has_node(supertype.name):
                    nxg.add_node(supertype.name, type='class')
                nxg.add_edge(classifier.name, supertype.name, type='generalization')
                count += 1
            
            for reference in classifier.eReferences:
                try:
                    if reference.eType is not None and not nxg.has_edge(classifier.name, reference.eType.name):
                        nxg.add_edge(
                            classifier.name, reference.eType.name, name=reference.name, \
                                type='reference' if reference.containment else 'association'
                        )
                except Exception as e:
                    pass
    

    for node in list(nxg.nodes()):
        if 'type' not in nxg.nodes[node]:
            nxg.remove_node(node)
    
    
    if all([nxg.nodes[node]['type'] == 'class' for node in nxg.nodes()]):
        print("Total nodes: ", nxg.number_of_nodes())
    
    if all([nxg.edges[edge]['type'] in ['generalization', 'association', 'reference'] for edge in nxg.edges()]):
        print("Total edges: ", nxg.number_of_edges())
    
    try:
        assert all(['type' in nxg.edges[edge] for edge in nxg.edges()])
    except AssertionError as e:
        print("Error in creating graph")
        exit(0)
    
    gen_edges = sum(1 for edge in nxg.edges() if nxg.edges[edge]['type']  == 'generalization')
    print("3 Total generalization edges: ", gen_edges)

    return nxg
        

def get_graph_from_files(file_names):
    graphs = list()
    count = 0
    for file_name in stqdm(file_names, desc="Creating Ecore Graphs"):
        graph = create_nx_from_ecore(file_name)

        if graph is not None:
            graphs.append((file_name, graph))
            
    print("Total files that could not be parsed: ", count)
    print("Total graphs: ", len(graphs))
    return graphs


def get_graphs_from_directory(dir):
    f_name, dir_name = os.path.basename(dir), os.path.dirname(dir)
    graph_pickle_file_name = os.path.join(dir_name, f_name + "_graphs.pkl")
    print("Graph pickle", graph_pickle_file_name)
    if os.path.exists(graph_pickle_file_name):
        print("Graphs found in cache!")
        graphs = pickle.load(open(os.path.join(dir_name, f_name + "_graphs.pkl"), 'rb'))
    else:
        file_names = find_files_with_extension(dir, 'ecore')
        graphs = get_graph_from_files(file_names)
        pickle.dump(graphs, open(graph_pickle_file_name, 'wb'))
    
    return graphs



def get_graphs_from_metadata(models_metadata, dir=None):
    graphs = list()
    models = models_metadata if dir is None else [os.path.join(dir, model) for model in models_metadata.keys()]
    graphs = get_graph_from_files(models)
    return graphs


def graph2str(g):
    return str(g.edges())


def remove_duplicates(graphs):
    return list({graph2str(g):g for g in graphs}.values())


def filter_graphs(graphs, min_edges=10):
    return [g for g in filter(lambda g: g.number_of_edges() >= min_edges, graphs)]


def clean_graph_set(graphs):
    graphs = remove_duplicates(graphs)
    graphs = filter_graphs(graphs)
    return graphs


def write_graphs_to_file(graphs, file_name):
    with open(file_name, 'wb') as f:
        pickle.dump(graphs, f)


def read_graphs_from_file(file_name):
    with open(file_name, 'rb') as f:
        graphs = pickle.load(f)
    return graphs


def write_clean_graphs_to_file(graphs, file_name):
    graphs = clean_graph_set(graphs)
    print("Total clean graphs: ", len(graphs))
    write_graphs_to_file(graphs, file_name)


def read_clean_graphs_from_file(file_name):
    graphs = read_graphs_from_file(file_name)
    graphs = clean_graph_set(graphs)
    return graphs

# if __name__ == '__main__':
#     dir_name = "datasets/test_data.zip"
#     graphs = get_graphs_from_zip_file(dir_name)
#     write_clean_graphs_to_file(graphs, dir_name.split(".")[0] + "_graphs.pkl")