from tqdm.auto import tqdm
import torch
import math
import os
from transformers import TrainingArguments
from transformers import AutoModelForSequenceClassification
from transformers import Trainer
import transformers

from transformers import AutoTokenizer
from models import TripleClassifier
from trainers import LMTrainer
from torch.utils.data import DataLoader
from parameters import parse_args
from graph_utils import get_graph_data
from data_utils import TripleDataset, TaskTypeDataset, get_dataset
from data_utils import compute_metrics

from data_utils import get_recommendation_metrics


def get_eval_stats(eval_result):
    stats = {
        'loss': eval_result['eval_loss'], 
        'perplexity': math.exp(eval_result['eval_loss']), 
        'accuracy': eval_result['eval_accuracy'],
    }
    return stats


def get_metrics_dataloader(model, dataloader):
    model.eval()
    with torch.no_grad():
        results = {k: 0 for k in ['MRR', 'Hits@1', 'Hits@3', 'Hits@5', 'Hits@10']}
        for batch in tqdm(dataloader, desc="Batches"):
            inputs = {k: v.to(model.device) for k, v in batch.items()}
            logits = model(**inputs).logits
            sp_metrics = get_recommendation_metrics(logits, inputs['labels'])

            for metric in results:
                results[metric] += sp_metrics[metric]
        
        for metric in results:
            results[metric] /= len(dataloader)
        
        return results



def get_metrics(model, dataset):
    test_dataloader = DataLoader(dataset['test'], batch_size=256, shuffle=False)
    unseen_dataloader = DataLoader(dataset['unseen'], batch_size=256, shuffle=False)

    test_results = get_metrics_dataloader(model, test_dataloader)
    unseen_results = get_metrics_dataloader(model, unseen_dataloader)

    results = {
        'test': test_results,
        'unseen': unseen_results,
    }

    print("Test Results")
    print(test_results)

    print("Unseen Results")
    print(unseen_results)

    return results
    


def fine_tune_hugging_face(dataset, tokenizer, args):

    batch_size = args.batch_size
    model = AutoModelForSequenceClassification.from_pretrained(pretrained_model_name_or_path=args.model_name, num_labels=args.num_labels, ignore_mismatched_sizes=True)
    model.resize_token_embeddings(len(tokenizer))
    
    logging_steps = len(dataset['train']) // batch_size
    print(f"Using model...{args.model_name}")
    model.resize_token_embeddings(len(tokenizer))
    print("Finetuning model...")
    training_args = TrainingArguments(
        output_dir='models',
        overwrite_output_dir=True,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        learning_rate=2e-5,
        weight_decay=0.01,
        warmup_steps=args.warmup_steps,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        fp16=True,
        logging_steps=logging_steps,
        num_train_epochs=1,
        save_total_limit=2,
        load_best_model_at_end=True,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset['train'],
        eval_dataset=dataset['test'],
        tokenizer=tokenizer,
    )
    for cb in trainer.callback_handler.callbacks:
        if isinstance(cb, transformers.integrations.NeptuneCallback):
            trainer.callback_handler.remove_callback(cb)
    

    print("Begin training...")
    results = dict()
    for i in tqdm(range(args.num_epochs), desc="Epochs"):

        trainer.train()
        trainer.save_model()
        eval_results = trainer.evaluate()
        print(f">>> Accuracy (after training): {eval_results['eval_accuracy']:.2f}")
        results[f'test (after)_{i}'] = {
            'accuracy_metrics': get_eval_stats(eval_results),
            'recommendation_metrics': get_metrics(trainer.model, dataset),
        }


        trainer.eval_dataset = dataset['unseen']
        eval_results = trainer.evaluate()
        results[f'unseen (after)_{i}'] = {
            'accuracy_metrics': get_eval_stats(eval_results),
            'recommendation_metrics': get_metrics(trainer.model, dataset),
        }
    
    return results


def fine_tune_manual(dataset, args):
    train_dataloader = DataLoader(dataset['train'], batch_size=args.batch_size, shuffle=True)
    test_dataloader = DataLoader(dataset['test'], batch_size=args.batch_size, shuffle=False)
    unseen_dataloader = DataLoader(dataset['unseen'], batch_size=args.batch_size, shuffle=False)

    model = TripleClassifier(
        num_labels=train_dataset.num_labels, 
        num_spt_labels=train_dataset.num_super_type_labels, 
        mask_token=tokenizer.mask_token, 
        mask_token_id=tokenizer.mask_token_id, 
        model_name=args.model_name
    )

    trainer = LMTrainer(
        model=model, 
        train_dataloader=train_dataloader, 
        test_dataloader=test_dataloader, 
        multi_label=args.multi_label, 
        num_epochs=args.num_epochs, 
        alpha=args.alpha
    )


    train_results = trainer.train()
    print("Evaluation on test set")
    test_results = trainer.test(args.num_epochs)

    print("Evaluation on unseen set")
    unseen_results = trainer.test(args.num_epochs, unseen_dataloader)
    
    return train_results, test_results, unseen_results


if __name__ == '__main__':
    args = parse_args()
    data_dir = args.data_dir
    args.graphs_file = os.path.join(data_dir, args.graphs_file)
    

    graph_data = get_graph_data(args.graphs_file)
    label_map, super_type_map = graph_data['entities_encoder'], graph_data['super_types_encoder']
    inverse_label_map = {v: k for k, v in label_map.items()}
    inverse_super_type_map = {v: k for k, v in super_type_map.items()}

    
    for i, data in enumerate(get_dataset(graph_data)):
        tokenizer = AutoTokenizer.from_pretrained(args.model_name)

        train_dataset = TripleDataset(data['train'], label_map, super_type_map, tokenizer, args.multi_label)
        test_dataset = TripleDataset(data['test'], label_map, super_type_map, tokenizer, args.multi_label)
        unseen_dataset = TripleDataset(data['unseen'], label_map, super_type_map, tokenizer, args.multi_label)

        # train_entity_dataset = TaskTypeDataset(train_dataset[:], task_type='entity')
        # test_entity_dataset = TaskTypeDataset(test_dataset[:], task_type='entity')
        # unseen_entity_dataset = TaskTypeDataset(unseen_dataset[:], task_type='entity')

        train_spt_dataset = TaskTypeDataset(train_dataset[:], task_type='super_type')
        test_spt_dataset = TaskTypeDataset(test_dataset[:], task_type='super_type')
        unseen_spt_dataset = TaskTypeDataset(unseen_dataset[:], task_type='super_type')

        # train_results, test_results, unseen_results = fine_tune_manual(
        #     {
        #         'train': train_dataset,
        #         'test': test_dataset,
        #         'unseen': unseen_dataset,
        #     },
        #     args
        # )

        # args.num_labels = len(label_map)
        # entity_classification_results = fine_tune_hugging_face(
        #     {
        #         'train': train_entity_dataset,
        #         'test': test_entity_dataset,
        #         'unseen': unseen_entity_dataset,
        #     },
        #     tokenizer,
        #     args
        # )
        
        args.num_labels = len(super_type_map)
        super_type_classification_results = fine_tune_hugging_face(
            {
                'train': train_spt_dataset,
                'test': test_spt_dataset,
                'unseen': unseen_spt_dataset,
            },
            tokenizer,
            args
        )

        # print("Entity Classification Results")
        # print(entity_classification_results)

        print("Super Type Classification Results")
        print(super_type_classification_results)

        break