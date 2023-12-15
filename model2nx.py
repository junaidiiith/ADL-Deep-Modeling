from pyecore.resources import ResourceSet, URI
import networkx as nx

import fnmatch
import os
import xmltodict
import json
from tqdm.auto import tqdm
import networkx as nx


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


extra_properties = [
    "isAbstract", "isDerived", "isDisjoint", "type", "isComplete", "isPowertype", "isExtensional", "isOrdered", "aggregationKind",
]

ecore_data = "umlEcore"
ontouml_models = "ontoumlJson"



documented_stereotypes = [
    'kind', 'subkind', 'phase', 'role', 'collective', 'relator', 'category', \
    'phasemixin', 'rolemixin', 'mixin', 'mode', 'quality', \
    'formal', 'material', 'mediation', 'characterization', 'derivation', 'structuration', \
    'componentof', 'memberof', 'subcollectionof', 'part-whole', 'containment', 'subquantityof'
]

exclude_limit = 50
exclude_fn, include_fn = lambda x: x[1][0] < exclude_limit, lambda x: x[1][0] >= exclude_limit
synonymous_mappings = {
    "humanagent": "agent",
    "institutionalagent": "agent",
    "complexevent": "event",
    "complexaction": "action",
    "partof": "part-whole",
    "part-of": "part-whole",
    "material relation": "material",
    "atomic event": "event",
    "historicalrolemixin": "rolemixin",
    "historicalrole": "role",
}


def xml_to_json(xml_string):
    xml_dict = xmltodict.parse(xml_string)
    json_data = json.dumps(xml_dict, indent=4)
    return json_data



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
            
            for reference in classifier.eReferences:
                try:
                    if reference.eType is not None and not nxg.has_edge(classifier.name, reference.eType.name):
                        nxg.add_edge(
                            classifier.name, reference.eType.name, name=reference.name, \
                                type='reference' if reference.containment else 'association'
                        )
                except Exception as e:
                    # print("ref", reference)
                    # raise(e)
                    pass
        
    return nxg


def get_graphs_from_dir(models_metadata, dir=None):
    graphs = list()
    count = 0
    models = models_metadata if dir is None else [os.path.join(dir, model) for model in models_metadata.keys()]
    for model_file_name in tqdm(models):
        try:
            g = create_nx_from_ecore(model_file_name)
            if g is not None:
                graphs.append(g)
        except Exception as e:
            print(model_file_name)
            count += 1
    print(count)
    return graphs

# if __name__ == '__main__':
#     from parameters import parse_args
#     args = parse_args()
#     models_dir = args.models_dir
#     min_stereotype_nodes = args.min_stereotype_nodes
#     ontouml_graphs = get_ontouml_to_nx(models_dir, min_stereotype_nodes)


