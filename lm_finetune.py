import os
from transformers import AutoTokenizer
from models import TripleClassifier
from sklearn.model_selection import StratifiedKFold
from trainers import Trainer
from torch.utils.data import DataLoader
from parameters import parse_args
from graph_utils import get_node_triples
from data_utils import TripleDataset




def get_dataset(node_triples, seed=42, test_size=0.1):
    X = [1]*len(node_triples)
    k_folds = int(1/test_size)
    i = 0
    skf = StratifiedKFold(n_splits=k_folds, shuffle=True, random_state=seed)
    for train_idx, test_idx in skf.split(X, X):
        train = [node_triples[i] for i in train_idx]
        test = [node_triples[i] for i in test_idx]

        # train, test = train_test_split(node_triples, test_size=test_size, random_state=seed)
        print(len(train), len(test))

        train_entities = set([t[0] for t in train] + [t[2] for t in train])
        test_seen = list([v for v in test if v[0] in train_entities])
        test_unseen = list([v for v in test if v[0] not in train_entities])

        seen_entities = list(set([t[0] for t in train] + [t[0] for t in test_seen]))
        unseen_entities = list(set([t[0] for t in test_unseen]))

        seen_super_types = list(set([st for t in train for st in t[3].split(', ')] + [st for t in test_seen for st in t[3].split(', ')]))
        unseen_super_types = list(set([st for t in test_unseen for st in t[3].split(', ')]))

        data = {
            'train': train,
            'test': test,
            'test_seen': test_seen,
            'test_unseen': test_unseen,
            'seen_entities': seen_entities,
            'unseen_entities': unseen_entities,
            'seen_super_types': seen_super_types,
            'unseen_super_types': unseen_super_types,
        }
        
        i += 1
        
        yield data


if __name__ == '__main__':
    args = parse_args()
    multi_label = args.multi_label
    data_dir = args.data_dir
    graphs_file = os.path.join(data_dir, args.graphs_file)
    model_name = args.model_name
    alpha = args.alpha
    num_epochs = args.num_epochs
    batch_size = args.batch_size

    node_triples = get_node_triples(graphs_file)

    for i, data in enumerate(get_dataset(node_triples)):
        label_map = {v: i for i, v in enumerate(data['seen_entities'])}
        stereotype_map = {v: i for i, v in enumerate(data['seen_super_types'])}
        tokenizer = AutoTokenizer.from_pretrained(model_name)

        train_dataset = TripleDataset(data['train'], label_map, stereotype_map, tokenizer, multi_label)
        test_dataset = TripleDataset(data['test_seen'], label_map, stereotype_map, tokenizer, multi_label)


        train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

        model = TripleClassifier(train_dataset.num_labels, train_dataset.num_stereotype_labels, model_name=model_name)

        trainer = Trainer(model, train_dataloader, test_dataloader, num_epochs=num_epochs, alpha=alpha)

        train_results = trainer.train()
        test_results = trainer.test(num_epochs)

        if i >= 3:
            break