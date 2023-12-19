import random
from collections import defaultdict
import fnmatch
import os
from sklearn.model_selection import StratifiedKFold
from tqdm.auto import tqdm
import json
import networkx as nx

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


def get_ontouml_to_nx(data_dir, min_stereotypes=10):
    ontouml_graphs = list()
    models = find_files_with_extension(data_dir, "json")
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



def mask_graph(graph, stereotypes_classes, mask_prob=0.2, use_stereotypes=False, use_rel_stereotypes=False):
    all_stereotype_nodes = [node for node in graph.nodes if 'stereotype' in graph.nodes[node]\
         and graph.nodes[node]['stereotype'] in stereotypes_classes and has_neighbours_incl_incoming(graph, node)\
            and (True if use_rel_stereotypes else graph.nodes[node]['type'] == 'Class')]
    
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
    
    

def mask_graphs(graphs, stereotypes_classes, mask_prob=0.2, use_stereotypes=False, use_rel_stereotypes=False):
    masked, unmasked, total = 0, 0, 0
    # for graph, f_name in tqdm(graphs, desc='Masking graphs'):
    for graph, _ in graphs:
        mask_graph(graph, stereotypes_classes, mask_prob=mask_prob, use_stereotypes=use_stereotypes, use_rel_stereotypes=use_rel_stereotypes)
        masked += len([node for node in graph.nodes if 'masked' in graph.nodes[node] and graph.nodes[node]['masked']])
        unmasked += len([node for node in graph.nodes if 'masked' in graph.nodes[node] and not graph.nodes[node]['masked']])
        total += len([node for node in graph.nodes if 'masked' in graph.nodes[node]])
        
    ## % of masked nodes upto 2 decimal places
    print(f"Masked {round(masked/total, 2)*100}%")
    print(f"Unmasked {round(unmasked/total, 2)*100}%")

    print("Total masked nodes:", masked)
    print("Total unmasked nodes:", unmasked)



def has_neighbours_incl_incoming(graph, node):
    edges = list(graph.edges(node))
    edges += list(graph.in_edges(node))
    return len(edges) != 0


def get_graphs_data_kfold(args):
    ontology_graphs = get_ontouml_to_nx(args.data_dir)
    label_encoder = get_label_encoder(ontology_graphs, args.exclude_limit)
    stereotypes_classes = list(label_encoder.keys())
    X = [1]*len(ontology_graphs)
    k_folds = int(1/args.train_split)
    skf = StratifiedKFold(n_splits=k_folds, shuffle=True, random_state=args.seed)
    for train_idx, val_idx in skf.split(X, X):
        seen_graphs = [ontology_graphs[i] for i in train_idx]
        unseen_graphs = [ontology_graphs[i] for i in val_idx]

        mask_graphs(seen_graphs, stereotypes_classes,\
                    mask_prob=args.mask_prob, 
                    use_stereotypes=args.use_stereotypes,
                    use_rel_stereotypes=args.use_rel_stereotypes)
        mask_graphs(unseen_graphs, stereotypes_classes,\
                    mask_prob=args.mask_prob, \
                    use_stereotypes=args.use_stereotypes,
                    use_rel_stereotypes=args.use_rel_stereotypes)
        yield seen_graphs, unseen_graphs, label_encoder
