import random
from collections import defaultdict
import fnmatch
import os
from zipfile import ZipFile
from sklearn.model_selection import StratifiedKFold
import torch
from tqdm.auto import tqdm
import json
import networkx as nx
from tqdm.auto import tqdm
from collections import deque
import re
import itertools
from datasets import EncodingsDataset
from uml_data_generation import get_encoding_size


SEP = "<sep>"
e_s = {'rel': 'relates', 'gen': 'generalizes'}
remove_extra_spaces = lambda txt: re.sub(r'\s+', ' ', txt.strip())


extra_properties = [
    "isAbstract", "isDerived", "isDisjoint", "type", "isComplete", "isPowertype", "isExtensional", "isOrdered", "aggregationKind",
]

frequent_stereotypes = ['kind', 'subkind', 'phase', 'role', 'category', 'mixin', 'rolemixin', 'phasemixin']

ONTOUML_ELEMENT_ID = 'id'
ONTOUML_ELEMENT_TYPE = 'type'
ONTOUML_ELEMENT_NAME = 'name'
ONTOUML_ELEMENT_DESCRIPTION = 'description'

ONTOUML_GENERALIZATION = "Generalization"
ONTOUML_GENERALIZATION_GENERAL = "general"
ONTOUML_GENERALIZATION_SPECIFIC = "specific"
ONTOUML_GENERALIZATION_SET = "GeneralizationSet"
ONTOUML_GENERALIZATION_SET_GENERALIZATIONS = "generalizations"
ONTOUML_GENERALIZATION_SET_IS_DISJOINT = "isDisjoint"
ONTOUML_GENERALIZATION_SET_IS_COMPLETE = "isComplete"

ONTOUML_PROJECT = "Project"
ONTOUML_PROJECT_MODEL = "model"
ONTOUML_PROJECT_MODEL_CONTENTS = "contents"
ONTOUML_RELATION = "Relation"
ONTOUML_PROPERTIES = "properties"
ONTOUML_RELATION_PROPERTY_TYPE = "propertyType"
ONTOUML_STEREOTYPE = "stereotype"
ONTOUML_CLASS = "Class"
ONTOUML_ENUMERATION = "enumeration"
ONTOUML_CLASS_LITERALS = 'literals'
ONTOUML_PACKAGE = "Package"
ONTOUML_LITERAL = "Literal"




def find_files_with_extension(root_dir, extension):
    matching_files = []

    # Recursively search for files with the specified extension
    for root, _, files in os.walk(root_dir):
        for filename in fnmatch.filter(files, f'*.{extension}'):
            matching_files.append(os.path.join(root, filename))

    return matching_files



def ontouml_id2obj(o_ontouml, id2obj_map):
    assert isinstance(o_ontouml, dict)
    for key in o_ontouml:
        if key == ONTOUML_ELEMENT_ID and ONTOUML_ELEMENT_TYPE in o_ontouml and o_ontouml[ONTOUML_ELEMENT_TYPE]\
              in [ONTOUML_CLASS, ONTOUML_RELATION, ONTOUML_GENERALIZATION_SET, ONTOUML_GENERALIZATION]\
                and ONTOUML_ELEMENT_DESCRIPTION in o_ontouml:
            id2obj_map[o_ontouml[ONTOUML_ELEMENT_ID]] = o_ontouml
        elif isinstance(o_ontouml[key], dict):
            ontouml_id2obj(o_ontouml[key], id2obj_map)
        elif isinstance(o_ontouml[key], list):
            for item in o_ontouml[key]:
                assert not isinstance(item, list)
                if isinstance(item, dict):
                    ontouml_id2obj(item, id2obj_map)




def get_nxg_from_ontouml_map(ontouml_id2obj_map, f_name='out.txt', directed=True):
    fp = open(f_name, 'w')
    g = nx.Graph() if not directed else nx.DiGraph()

    for k, v in ontouml_id2obj_map.items():
        node_name = v[ONTOUML_ELEMENT_NAME] if (ONTOUML_ELEMENT_NAME in v and v[ONTOUML_ELEMENT_NAME] is not None) else 'Null'
        
        if v[ONTOUML_ELEMENT_TYPE] in [ONTOUML_CLASS, ONTOUML_RELATION]:
            g.add_node(k, name=node_name, type=v[ONTOUML_ELEMENT_TYPE], description='')
            for prop in extra_properties:
                g.nodes[k][prop] = v[prop] if prop in v else False

            fp.write(f"Node: {node_name} type: {v[ONTOUML_ELEMENT_TYPE]}\n")
        
        fp.write(f"Node: {node_name} type: {v[ONTOUML_ELEMENT_TYPE]}\n")
        if ONTOUML_STEREOTYPE in v and v[ONTOUML_STEREOTYPE] is not None:
            g.nodes[k][ONTOUML_STEREOTYPE] = v[ONTOUML_STEREOTYPE].lower()
            fp.write(f"\tONTOUML_STEREOTYPE: {v[ONTOUML_STEREOTYPE].lower()}\n")
        

        if ONTOUML_ELEMENT_DESCRIPTION in v and v[ONTOUML_ELEMENT_DESCRIPTION] is not None:
            fp.write(f"Description: {v[ONTOUML_ELEMENT_DESCRIPTION]}\n")
            g.nodes[k][ONTOUML_ELEMENT_DESCRIPTION] = v[ONTOUML_ELEMENT_DESCRIPTION]
        

        if v[ONTOUML_ELEMENT_TYPE] == ONTOUML_CLASS:
                if ONTOUML_CLASS_LITERALS in v and v[ONTOUML_CLASS_LITERALS] is not None:
                    literals = v[ONTOUML_CLASS_LITERALS] if isinstance(v[ONTOUML_CLASS_LITERALS], list) else [v[ONTOUML_CLASS_LITERALS]]
                    literals_str = ", ".join([literal[ONTOUML_ELEMENT_NAME] for literal in literals])
                    g.nodes[k][ONTOUML_PROPERTIES] = literals_str
                    fp.write(f"\tLiterals: {literals_str}\n")
            
                elif ONTOUML_PROPERTIES in v and v[ONTOUML_PROPERTIES] is not None:
                    properties = v[ONTOUML_PROPERTIES] if isinstance(v[ONTOUML_PROPERTIES], list) else [v[ONTOUML_PROPERTIES]]
                    properties_str = ", ".join([property[ONTOUML_ELEMENT_NAME] for property in properties])
                    g.nodes[k][ONTOUML_PROPERTIES] = properties_str
                    fp.write(f"\tProperties: {properties_str}\n")
            

        elif v[ONTOUML_ELEMENT_TYPE] == ONTOUML_RELATION:    
            properties = v[ONTOUML_PROPERTIES] if isinstance(v[ONTOUML_PROPERTIES], list) else [v[ONTOUML_PROPERTIES]]
            assert len(properties) == 2
            try:
                x_id = properties[0][ONTOUML_RELATION_PROPERTY_TYPE][ONTOUML_ELEMENT_ID]
                y_id = properties[1][ONTOUML_RELATION_PROPERTY_TYPE][ONTOUML_ELEMENT_ID]
                x_name = ontouml_id2obj_map[x_id][ONTOUML_ELEMENT_NAME] if ONTOUML_ELEMENT_NAME is not None else ''
                y_name = ontouml_id2obj_map[y_id][ONTOUML_ELEMENT_NAME] if ONTOUML_ELEMENT_NAME is not None else ''

                g.add_edge(x_id, v[ONTOUML_ELEMENT_ID], type='rel')
                g.add_edge(v[ONTOUML_ELEMENT_ID], y_id, type='rel')
                fp.write(f"\tRelationship:, {x_name} --> {y_name}\n")
            except TypeError as e:
                # print(f"Error in {v[ONTOUML_ELEMENT_TYPE]}, {v[ONTOUML_ELEMENT_NAME]}")
                pass

        
        elif v[ONTOUML_ELEMENT_TYPE] == ONTOUML_GENERALIZATION:
            general = v[ONTOUML_GENERALIZATION_GENERAL][ONTOUML_ELEMENT_ID]
            specific = v[ONTOUML_GENERALIZATION_SPECIFIC][ONTOUML_ELEMENT_ID]
            general_name = ontouml_id2obj_map[general][ONTOUML_ELEMENT_NAME]\
                  if ONTOUML_ELEMENT_NAME in ontouml_id2obj_map[general] else ''
            specific_name = ontouml_id2obj_map[specific][ONTOUML_ELEMENT_NAME] \
                  if ONTOUML_ELEMENT_NAME in ontouml_id2obj_map[specific] else ''
            fp.write(f"\tGeneralization:, {specific_name} -->> {general_name}\n")
            g.add_edge(specific, general, type='gen')

    
    return g


def get_all_files(zip_file):
    """
        Unzip data_dir zip files and get all the JSON files
    """

    with ZipFile(zip_file, 'r') as zip:
        zip.extractall()
        all_files = list()
        for root, _, files in os.walk(zip_file.name.split('.')[0]):
            for file in files:
                all_files.append(os.path.join(root, file))
        
    return all_files


def get_ontouml_to_nx(data_dir, min_stereotypes=10):
    ontouml_graphs = list()
    models = find_files_with_extension(data_dir, "json")
    print(data_dir, len(models))
        
    for mfp in tqdm(models, desc=f"Reading {len(models)} OntoUML models"):
        if mfp.endswith(".ecore") or mfp.endswith(".json"):
            json_obj = json.loads(open(mfp, 'r', encoding='iso-8859-1').read())
            id2obj_map = {}
            ontouml_id2obj(json_obj, id2obj_map)
            g = get_nxg_from_ontouml_map(id2obj_map, mfp.replace(".json", ".txt"))
            stereotype_nodes = [node for node, stereotype in g.nodes(data=ONTOUML_STEREOTYPE) if stereotype is not None]
            if len(stereotype_nodes) >= min_stereotypes:
                ontouml_graphs.append((g, mfp))
    
    return ontouml_graphs


def get_label_encoder(graphs, exclude_limit):
    stereotypes = defaultdict(int)
    for g, _ in graphs:
        for node in g.nodes:
            if 'stereotype' in g.nodes[node]:
                stereotypes[g.nodes[node]['stereotype']] += 1


    if exclude_limit != -1:
        stereotypes_classes = [stereotype for stereotype, _ in filter(lambda x: x[1] > exclude_limit, stereotypes.items())]
    else:
        stereotypes_classes = [stereotype for stereotype, _ in filter(lambda x: x[0] in frequent_stereotypes, stereotypes.items())]
    # print(len(stereotypes_classes))
    label_encoder = {label: i for i, label in enumerate(stereotypes_classes)}
    return label_encoder



def mask_graph(graph, stereotypes_classes, mask_prob=0.2, use_stereotypes=True):
    all_stereotype_nodes = [node for node in graph.nodes if 'stereotype' in graph.nodes[node]\
         and graph.nodes[node]['stereotype'] in stereotypes_classes and has_neighbours_incl_incoming(graph, node)]
    
    assert all(['stereotype' in graph.nodes[node] for node in all_stereotype_nodes]), "All stereotype nodes should have stereotype property"

    total_masked_nodes = int(len(all_stereotype_nodes) * mask_prob)
    masked_nodes = random.sample(all_stereotype_nodes, total_masked_nodes)
    unmasked_nodes = [node for node in all_stereotype_nodes if node not in masked_nodes]

    for node in masked_nodes:
        graph.nodes[node]['masked'] = True
        graph.nodes[node]['use_stereotype'] = False
    
    for node in unmasked_nodes:
        graph.nodes[node]['masked'] = False
        graph.nodes[node]['use_stereotype'] = use_stereotypes

    assert all(['masked' in graph.nodes[node] for node in all_stereotype_nodes]), "All stereotype nodes should be masked or unmasked"
    
    

def mask_graphs(graphs, stereotypes_classes, mask_prob=0.2):
    masked, unmasked, total = 0, 0, 0
    # for graph, f_name in tqdm(graphs, desc='Masking graphs'):
    for graph, _ in graphs:
        mask_graph(graph, stereotypes_classes, mask_prob=mask_prob)
        masked += len([node for node in graph.nodes if 'masked' in graph.nodes[node] and graph.nodes[node]['masked']])
        unmasked += len([node for node in graph.nodes if 'masked' in graph.nodes[node] and not graph.nodes[node]['masked']])
        total += len([node for node in graph.nodes if 'masked' in graph.nodes[node]])
        
    ## % of masked nodes upto 2 decimal places
    print(f"Masked {round(masked/total, 2)*100}%")
    print(f"Unmasked {round(unmasked/total, 2)*100}%")

    # print("Total masked nodes:", masked)
    # print("Total unmasked nodes:", unmasked)



def has_neighbours_incl_incoming(graph, node):
    edges = list(graph.edges(node))
    edges += list(graph.in_edges(node))
    return len(edges) != 0


def get_graphs_data_kfold(args):
    ontology_graphs = get_ontouml_to_nx(args.data_dir)
    label_encoder = get_label_encoder(ontology_graphs, args.exclude_limit)
    stereotypes_classes = list(label_encoder.keys())
    X = [1]*len(ontology_graphs)
    # k_folds = int(1/args.test_size)
    k_folds = 5
    skf = StratifiedKFold(n_splits=k_folds, shuffle=True, random_state=args.seed)
    for train_idx, val_idx in skf.split(X, X):
        seen_graphs = [ontology_graphs[i] for i in train_idx]
        unseen_graphs = [ontology_graphs[i] for i in val_idx]

        mask_graphs(seen_graphs, stereotypes_classes,\
                    mask_prob=args.ontouml_mask_prob)
        mask_graphs(unseen_graphs, stereotypes_classes,\
                    mask_prob=args.ontouml_mask_prob)
        yield seen_graphs, unseen_graphs, label_encoder


def check_stereotype_relevance(g, n):
    return 'use_stereotype' in g.nodes[n] and g.nodes[n]['use_stereotype']

def process_name_and_steroetype(g, n):
    string = g.nodes[n]['name'] if g.nodes[n]['name'] != "Null" else ""
    string += f' {g.nodes[n]["stereotype"]} ' if check_stereotype_relevance(g, n) else ""
        
    return string

def process_node_for_string(g, n, src=True):
    if g.nodes[n]['type'] == 'Class':
        return [process_name_and_steroetype(g, n)]
        
    strings = list()
    node_str = process_name_and_steroetype(g, n)
    edges = list(g.in_edges(n)) if src else list(g.out_edges(n))
    for edge in edges:
        v = edge[0] if src else edge[1]
        v_str = f" {process_edge_for_string(g, edge)} {process_name_and_steroetype(g, v)}"
        n_str = v_str + node_str if src else node_str + v_str
        strings.append(n_str)
    return list(set(map(remove_extra_spaces, strings)))


def process_edge_for_string(g, e):
    edge_type_s = e_s[g.edges()[e]['type']]
    return remove_extra_spaces(f" {edge_type_s} ")


def get_triples_from_edges(g, edges=None):
    if edges is None:
        edges = g.edges()
    triples = []
    for edge in edges:
        u, v = edge
        edge_str = process_edge_for_string(g, edge)
        u_strings, v_strings = process_node_for_string(g, u, src=True), process_node_for_string(g, v, src=False)
        for u_str, v_str in itertools.product(u_strings, v_strings):
            pos_triple = u_str + f" {edge_str} " + v_str
            triples.append(remove_extra_spaces(pos_triple))

    return triples

def process_path_string(g, path):
    edges = list(zip(path[:-1], path[1:]))
    triples = get_triples_from_edges(g, edges)
    
    return remove_extra_spaces(f" {SEP} ".join(triples))


def get_triples_from_node(g, n, distance=1):
    triples = list()
    use_stereotype = g.nodes[n]['use_stereotype'] if 'use_stereotype' in g.nodes[n] else False
    g.nodes[n]['use_stereotype'] = False
    node_neighbours = get_node_neighbours(g, n, distance)
    for neighbour in node_neighbours:
        paths = [p for p in nx.all_simple_paths(g, n, neighbour, cutoff=distance)]
        for path in paths:
            triples.append(process_path_string(g, path))
    
    g.nodes[n]['use_stereotype'] = use_stereotype
    return triples


def get_node_str(g, n, distance=1):
    node_triples = get_triples_from_node(g, n, distance)
    return remove_extra_spaces(f" | ".join(node_triples))


def find_nodes_within_distance(graph, start_node, distance):
    q, visited = deque(), dict()
    q.append((start_node, 0))
    
    while q:
        n, d = q.popleft()
        if d <= distance:
            visited[n] = d
            neighbours = [neighbor for neighbor in graph.neighbors(n) if neighbor != n and neighbor not in visited]
            for neighbour in neighbours:
                if neighbour not in visited:
                    q.append((neighbour, d + 1))
    
    sorted_list = sorted(visited.items(), key=lambda x: x[1])
    return sorted_list


def get_node_neighbours(graph, start_node, distance):
    neighbours = find_nodes_within_distance(graph, start_node, distance)
    max_distance = max(distance for _, distance in neighbours)
    distance = min(distance, max_distance)
    return [node for node, d in neighbours if d == distance]



def get_triples(graphs, distance=1, train=True):
    triples = list()
    for g, _ in tqdm(graphs):
        triples += get_graph_triples(g, distance=distance, train=train)
    return triples


def get_graph_triples(g, distance=1, train=True):
    relevant_nodes = [node for node in g.nodes if 'masked' in g.nodes[node] and g.nodes[node]['masked'] != train]
    node_strings = [get_node_str(g, node, distance) for node in relevant_nodes]
    node_triples = list()
    for node, node_str in zip(relevant_nodes, node_strings):
        name = g.nodes[node]['name']
        node_type = g.nodes[node]['type']
        if node_str == "":
            node_str = name
        label_str = g.nodes[node]['stereotype']
        # prompt_str = f"{node_type}"
        prompt_str = f"{node_type} {name}: {node_str}"
        node_triples.append((prompt_str, label_str))
    return node_triples


def get_triples_dataset(triples, label_encoder, tokenizer):
    max_length = get_encoding_size(triples, tokenizer)
    max_length = max_length if max_length < 512 else 512
    inputs, labels = [i[0] for i in triples], [label_encoder[i[1]] for i in triples]
    input_encodings = tokenizer(inputs, truncation=True, padding=True, max_length=max_length, return_tensors='pt')
    input_encodings['labels'] = torch.tensor(labels)
    dataset = EncodingsDataset(input_encodings)
    return dataset