# Deep Modeling for UML

This project applies transformers-based language modeling and graph neural network-based link prediction techniques to assist UML modelers. The objective is to leverage natural language labels in UML models and the graph structure of UML models to learn patterns for various tasks.

## Tasks

### 1. Partial Model Completion 
   - a. Given a UML class and its relations with neighboring classes, predict the name of the abstract class.
   - b. Given the abstract classes and relations of a UML class, predict the name of the UML class.

### 2. Ontological Stereotype Prediction 
   - Predict the "stereotype" of a UML class in an OntoUML, considering the name of the class, its neighboring classes, and relationships.

### 3. Link Prediction 
   - Given a UML model with missing links between the classes, predict the missing links.

Partial model completion is achieved in two stages: Pretraining and Finetuning.

## Training Stages

### Pretraining Stage
In the pretraining stage, a `UMLGPT` model is trained on graph data as a next token prediction task. Nodes are represented as strings in the format `<node name> <relations> <super types>`. The model learns generalized patterns in the UML models data.

### Finetuning Stage
The trained `UMLGPT` model is then used for a sequence classification task using the `UMLGPTClassifier` class. Given the string representation of a node, the model is finetuned to predict the super type of the node or the node name.

For example, in the abstract class prediction task, the model predicts the abstract class label of the UML class based on the `[entity relations]` string representation.

## Tokenization

Language models tokenize strings into vectors of numbers using tokenizers. This project supports both pretrained language model (PLM) tokenizers and a custom `VocabTokenizer` class based on node strings data.

The `VocabTokenizer` class is implemented similarly to the `AutoTokenizer` class of Hugging Face to maintain a consistent API.

The `UMLGPTTrainer` class implements a trainer that uses a `UMLGPT` or `UMLGPTClassifier` model and trains the model for next token prediction or a sequence classification task.

### OntoUML Stereotype Prediction

This task employs the same classes (`UMLGPT` or `UMLGPTClassifier`) as the first task to predict the UML class stereotype.

### Link Prediction

The Link Prediction task uses node embeddings learned from the pretraining phase in the `UMLGPT` model. The `GNNModel` utilizes these node embeddings and is trained along with a Multi-Layer Perceptron Predictor to detect if a link exists between two nodes or not.

## Metrics

For the first two tasks (sequence classification), the chosen metrics are:
1. MRR - Mean Reciprocal Rank
2. Hits@k with k = 1, 3, 5, 10
3. Precision

For the third Link Prediction task, the metric chosen is `roc_auc_score`.

| Task No. | Metric Name    | Target | Achieved | Based on                                           |
|----------|-----------------|--------|----------|-----------------------------------------------------|
| 1        | Hits@10         | 0.48   | 0.496    | Bert Sequence Classifier                            |
| 2        | Precision       | 0.64   | 0.83     | [1](https://link.springer.com/chapter/10.1007/978-3-031-34560-9_17) and [2](https://www.researchgate.net/profile/Tiago-Prince-Sales/publication/370527243_Inferring_Ontological_Categories_of_OWL_Classes_Using_Foundational_Rules/links/6454ac435762c95ac3742e38/Inferring-Ontological-Categories-of-OWL-Classes-Using-Foundational-Rules.pdf) |
| 3        | roc_auc_score   | 0.860  | 0.829    | GPT2-based embeddings as opposed to UMLGPT2 embeddings |

For first case, note that these work have the same problem statement, i.e., to predict the class name given the partial model [EClassifier Recall@10 score - 0.4258](https://link.springer.com/article/10.1007/s10270-021-00929-3), [Local Context Recall@10 score - 0.458](https://link.springer.com/article/10.1007/s10270-021-00929-3), however the recall score cannot provide a valid comparison because they have not made the data public. 

For the second case, one of the baselines is my previous work that used word2vec and GNN based node classification to predict the OntoUML stereotype. The second work is a ontological rules based approach to infer the stereotypes. In case of the rule based approach, their hits@1 score is fairly poor - less than 30%, however their hits@3 score is better. However in both case, my approach using embeddings from pretrained language model and finetuning the embeddings for sequence classification task outperform both the works.

For the last case, I could not find any baseline as far as academia is concerned. However, I considered a comparison of embeddings from language models i.e., on one hand, embeddings from a pretrained language model and on the other, embeddings from the UMLGPT model implemented in this project. Both these embeddings are then used to train the GNN model as implemented in this project and it turns out the pretrained embeddings marginally outperform the UMLGPT model.



### File Descriptions

#### [data_generation_utils.py](data_generation_utils.py)
This file encapsulates essential methods and classes for creating PyTorch datasets, extracting data from UML models, and converting them into strings.

#### [model2nx.py](model2nx.py)
This script processes the .ecore files of UML models dataset, converting and storing them in .pkl format.

#### [models.py](models.py)
This file encompasses all implemented PyTorch `nn.Module` classes, such as UMLGPT, UMLGPTClassifier, MLPPredictor, and GNNModel. Additionally, it includes `nn.Module` classes required to create transformer blocks.

#### [ontouml_data_utils.py](ontouml_data_utils.py)
This module houses all necessary methods and classes for extracting data from JSON files of OntoUML models within the `datasets` folder.

#### [parameters.py](parameters.py)
All command-line argument parameters are contained and explained in this file.

#### [pretraining.py](pretraining.py)
This script executes the pretraining phase on UML models data.

#### [uml_classification.py](uml_classification.py)
This script performs sequence classification on UML models, predicting the UML class name or the UML abstract class of a UML class. It supports specifying a pretrained tokenizer or a custom vocab tokenizer. The model for classification can be an untrained UMLGPT, a pretrained UMLGPT model, or any model from the Hugging Face library.

#### [utils.py](utils.py)
This file contains methods to calculate metrics on the predictions.

#### [ontouml_classification.py](ontouml_classification.py)
This script executes sequence classification on UML models to predict the OntoUML stereotype of a UML class or relation. The tokenizer is always from a pretrained language model, and a custom tokenizer is not yet implemented for this case. The model for classification can be an untrained UMLGPT, a pretrained UMLGPT model, or any model from the Hugging Face library. [Link to ontouml_classification.py](ontouml_classification.py)

#### [link_prediction.py](link_prediction.py)
This script is used to execute link prediction between graphs on UML models. The tokenizer is always from a pretrained language model, and a custom tokenizer is not yet implemented for this case. [Link to link_prediction.py](link_prediction.py)

#### [graph_utils.py](graph_utils.py)
This file is used to create node triples, i.e., UML class, relation, and abstract class triples for UML models.

#### [ontouml_data_utils.py](ontouml_data_utils.py)
This script is used to create node triples for OntoUML classes. In this case, the node triple contains information not only about the neighbors but up to a distance `d`, as specified by the argument `distance`.

#### [trainer.py](trainer.py)
This file specifies all the trainers for the three different tasks.


## Setting up to run

Install dependencies - The dependencies are available in requirements.txt file and can be install by executing - 

```bash
pip install -r requirements.txt
```

## Run Configurations

All Run Configurations for the three tasks:
- PLM = pretrained language model
- Word tokenizer = tokenizer generated from VocabTokenizer

All the parameters are explained in the [parameters.py](parameters.py) file.

### 1. Pretraining with PLM Tokenizer and UMLGPT

```bash
python pretraining.py --tokenizer=bert-base-cased --gpt_model=uml-gpt --num_layers=6 --num_heads=8 --embed_dim=256 --batch_size=128 --lr=0.0001 --num_epochs=1
```

### 2. Pretraining with word tokenizer and UMLGPT

```bash
python pretraining.py --tokenizer=word --gpt_model=uml-gpt --num_layers=6 --num_heads=8 --embed_dim=256 --batch_size=128 --lr=0.0001 --num_epochs=1
```

### 3. Pretraining with PLM

```bash
python pretraining.py --gpt_model=gpt2 --batch_size=128 --lr=0.00001 --num_epochs=2
```

### Super Type and Entity Classification

### 4. Classification directly with PLM Tokenizer and UMLGPT

```bash
python uml_classification.py --tokenizer=bert-base-cased --classification_model=uml-gpt --num_layers=6 --num_heads=8 --embed_dim=256 --batch_size=128 --lr=0.0001 --num_epochs=1 --class_type=super_type
```

```bash
python uml_classification.py --tokenizer=bert-base-cased --classification_model=uml-gpt --num_layers=6 --num_heads=8 --embed_dim=256 --batch_size=128 --lr=0.0001 --num_epochs=1 --class_type=entity
```

### 5. Classification directly with word Tokenizer and UMLGPT

```bash
python uml_classification.py --tokenizer=word --classification_model=uml-gpt --num_layers=6 --num_heads=8 --embed_dim=256 --batch_size=128 --lr=0.0001 --num_epochs=1 --class_type=super_type
```

```bash
python uml_classification.py --tokenizer=word --classification_model=uml-gpt --num_layers=6 --num_heads=8 --embed_dim=256 --batch_size=128 --lr=0.0001 --num_epochs=1 --class_type=entity
```

### 6. Classification directly with PLM

```bash
python uml_classification.py --classification_model=bert-base-cased --batch_size=128 --lr=0.00001 --num_epochs=1 --class_type=super_type
```

```bash
python uml_classification.py --classification_model=bert-base-cased --batch_size=128 --lr=0.00001 --num_epochs=1 --class_type=entity
```

### 7. Classification with pretrained UMLGPT and PLM Tokenizer

*Tokenizer used here should be same as the tokenizer used for pretraining*

```bash
python uml_classification.py --classification_model=uml-gpt --from_pretrained=models/pre_uml-gpt_tok=bert-base-cased/best_model.pt --tokenizer=bert-base-cased --batch_size=128 --lr=0.0001 --num_epochs=1 --class_type=super_type
```

```bash
python uml_classification.py --classification_model=uml-gpt --from_pretrained=models/pre_uml-gpt_tok=bert-base-cased/best_model.pt --tokenizer=bert-base-cased --batch_size=128 --lr=0.0001 --num_epochs=1 --class_type=entity
```

### 8. Classification with pretrained UMLGPT with word Tokenizer

*Tokenizer used here should be same as the tokenizer used for pretraining*

```bash
python uml_classification.py --classification_model=uml-gpt --from_pretrained=models/pre_uml-gpt_tok=word/best_model.pt --tokenizer=word --batch_size=128 --lr=0.0001 --num_epochs=1 --class_type=super_type
```

```bash
python uml_classification.py --classification_model=uml-gpt --from_pretrained=models/pre_uml-gpt_tok=word/best_model.pt --tokenizer=word --batch_size=128 --lr=0.0001 --num_epochs=1 --class_type=entity
```

### 9. Classification with pretrained PLM and PLM Tokenizer

```bash
python uml_classification.py --classification_model=uml-gpt --from_pretrained=models/pre_gpt2/best_model --tokenizer=word --batch_size=128 --lr=0.00001 --num_epochs=1 --class_type=super_type
```

```bash
python uml_classification.py --classification_model=uml-gpt --from_pretrained=models/pre_gpt2/best_model --tokenizer=word --batch_size=128 --lr=0.00001 --num_epochs=1 --class_type=entity
```

### 13. OntoUML Stereotype Classification

```bash
ontouml_classification.py --data_dir=datasets/ontoumlModels --classification_model=uml-gpt --num_layers=6 --num_heads=8 --embed_dim=256 --num_epochs=1
```

### Link Prediction

### 10. Link Prediction with pretrained UMLGPT and PLM Tokenizer

```bash
python link_prediction.py --from_pretrained=models/pre_uml-gpt_tok=bert-base-cased/best_model.pt --num_layers=2 --embed_dim=256 --tokenizer=bert-base-cased
```

### 12. Link Prediction with pretrained PLM and PLM Tokenizer

*Tokenizer used here should be same as the tokenizer used for pretraining*

```bash
python link_prediction.py --from_pretrained=models/pre_gpt2/best_model --num_layers=2 --embed_dim=256 --tokenizer=gpt2
```