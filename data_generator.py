import os
import random
import shutil
from model2nx import get_ontouml_to_nx
from nx2data import OntoUMLNodeClassificationDataset, GraphDataset
from nx2data import generate_paths_from_graph
from tqdm.auto import tqdm


def get_graphs(args):
    models_dir = args.data_dir
    min_stereotype_nodes = args.min_stereotype_nodes
    return get_ontouml_to_nx(models_dir, min_stereotype_nodes)


def get_node_classification_dataset(args):
    ontouml_graphs = get_graphs(args)
    print("Total graphs:", len(ontouml_graphs))
    ontouml_dataset = OntoUMLNodeClassificationDataset(ontouml_graphs, args)
    label_encoder = {c:i for i, c in enumerate(ontouml_dataset.allowed_stereotypes)}
    train_graphs = [ontouml_dataset[i] for i in ontouml_dataset.train_idx]
    test_graphs = [ontouml_dataset[i] for i in ontouml_dataset.test_idx]

    train_graph_dataset = GraphDataset(train_graphs, label_encoder)
    test_graph_dataset = GraphDataset(test_graphs, label_encoder)

    return train_graph_dataset, test_graph_dataset


def get_ontouml_dataset(args):
    ontouml_graphs = get_graphs(args)
    print("Total graphs:", len(ontouml_graphs))
    ontouml_dataset = OntoUMLNodeClassificationDataset(ontouml_graphs, args)

    return ontouml_dataset


def generate_model_generation_dataset(args, suffix=''):
    graphs = get_graphs(args)
    for graph, f_name in tqdm(graphs):
        f_name = f_name.split('/')[-1].split('.')[0]
        fp = args.save_dir + f'/{f_name}{suffix}'
        if not os.path.exists(fp):
            generate_paths_from_graph(graph, distance=args.distance, num_neighbours=args.lsr, fp=fp)


def get_model_generation_file(args):
    suffix = f"_d={args.distance}_lsr={args.lsr}.txt"
    model_gen_file = args.save_dir + f'/{suffix}{args.model_gen_file}'

    if not os.path.exists(model_gen_file):
        print("Data not present. Generating path data files...")
        generate_model_generation_dataset(args, suffix)
        tr = args.mgtr
        files = [args.save_dir + f'/{f_name}' for f_name in os.listdir(args.save_dir) if f_name.endswith(suffix)]
        print("Total files:", len(files))
        mod_gen_files = random.sample(files, int(tr * len(files)))
        os.makedirs(args.models_gen_dir, exist_ok=True)
        for file in mod_gen_files:
            shutil.copy(file, args.models_gen_dir)

        with open(model_gen_file, 'w') as f:
            for file in tqdm(mod_gen_files, desc=f"Merging {len(mod_gen_files)} files for model generation"):
                with open(file, 'r') as f2:
                    for line in f2.readlines():
                        f.write(line)
    
    return model_gen_file


# def main():
#     import dgl
#     args = parse_args()
#     train_graph_dataset, test_graph_dataset = get_node_classification_dataset(args)
#     print(train_graph_dataset, test_graph_dataset)

#     for graph in train_graph_dataset:
#         g = dgl.graph((graph['edge_index'][0], graph['edge_index'][1]))
#         assert len(graph['node_texts']) == g.number_of_nodes()


# main()