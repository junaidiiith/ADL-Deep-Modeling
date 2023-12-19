Deep Modeling for UML


This Project applies transformers based language modeling and graph neural network based link prediction techniques to assist UML modelers. The idea is to use the natural language labels in the UML models and the graph structure of UML models to learn patterns in order to acheive the following tasks -  

1. Partial model completion 
    a. given a UML class and its relations with the neighbouring classes, predict the name of the abstract class
    b. given the abstract classes and the relations of a UML class, predict the name of the UML class

2. Ontological stereotype prediction - The classes and relations in an OntoUML have a crucial meta-property called its ``stereotype``. The task is that given the name of a UML class and its neighbouring classes and relationships, predict the ``stereotype`` of the class - ontological enrichment of the class

3. Link Prediction - Given a UML model with missing links between the classes, predict the missing links

Partial model completion is acheived in two stages - Pretraining and finetuning

In the pretraining stage, a ``UMLGPT`` model is trained on the graph data as a next token prediction task. The way graph data is represented as a sequence is by represented each node as a string of format <node name> <relations> <super types>. The language model learn to predict the next token in the sequences generated for the nodes. In this stage, the model is supposed to learn overall generalized patterns in the UML models data.

In the finetuning stage, the trained ``UMLGPT`` model is used for a sequence classification task using ``UMLGPTClassifier`` class. Given the string representation of a node, the model is finetuned to predict the super type of the node or the node name. 

For e.g., in case of abstract class prediction -  Given the [entity relations] string representation of a UML class, the model aims to predict the abstract class label of the UML class. Similarly for the entity prediction task, given the [super_type relations] string representation of a UML class, the model is trained to predict the entity label of the UML class.  

Language models first tokenize the string i.e., represent them as vectors of numbers using tokenizers. However, this means that each word should be associated with the number. This can lead to a large vocubulary. Pretrained language model like BERT, GPT have tokenizers that can avoid such situations by breaking words into token. 
In this project, a tokenizer from a pretrained language model can be used and also a custom VocabTokenizer class is implemented that creates the tokenizer from the node strings data of the UML models. The VocabTokenizer class is implemented very similar to the AutoTokenizer class of hugging face in order to have a similar API.

``UMLGPTTrainer`` class implements a trainer that uses a ``UMLGPT`` or ``UMLGPTClassifier`` model and trains the model for next token prediction or a sequence classification task.


2. OntoUML stereotype prediction - This task is quite similar to the first task which uses the same classes i.e., ``UMLGPT`` or ``UMLGPTClassifier`` to predict the UML class stereotype. 

3. The Link Prediction task uses the node embeddings learn from the pretraining phase i.e., from ``UMLGPT`` model (or can use embeddings from the ``UMLGPTClassifier`` but this part is not implemented yet). Then the ``GNNModel`` uses the node embeddings and is trained along with a Multi Layer Perceptron Predictor to detect if a link exists between two nodes or not.


Metrics used for each task - 

The first two tasks are sequence classification tasks therefore, the chosen metrics to determine the quality of the predictions are -

1. MRR - Mean Reciprocal Rank
2. Hits@k with k = 1, 3, 5, 10

In case of the third link prediction task, the metric chosen is - roc_auc_score

Run Configs - 

All Run Configurations for the three task 

PLM = pretrained language model
word tokenizer = tokenizer generated from VocabTokenizer

All the parameters are explained in the [link to parameters.py] file.


1a. Pretraining with PLM Tokenizer and UMLGPT
python pretraining.py --tokenizer=bert-base-cased --gpt_model=uml-gpt --num_layers=6 --num_heads=8 --embed_dim=256 --batch_size=128 --lr=0.0001 --num_epochs=1

2. Pretraining with word tokenizer and UMLGPT
python pretraining.py --tokenizer=word --gpt_model=uml-gpt --num_layers=6 --num_heads=8 --embed_dim=256 --batch_size=128 --lr=0.0001 --num_epochs=1

3. Pretraining with PLM
python pretraining.py --gpt_model=gpt2 --batch_size=128 --lr=0.00001 --num_epochs=2




Super Type and Entity Classification


4. Classification directly with PLM Tokenizer and UMLGPT
python uml_classification.py --tokenizer=bert-base-cased --classification_model=uml-gpt --num_layers=6 --num_heads=8 --embed_dim=256 --batch_size=128 --lr=0.0001 --num_epochs=1 --class_type=super_type
python uml_classification.py --tokenizer=bert-base-cased --classification_model=uml-gpt --num_layers=6 --num_heads=8 --embed_dim=256 --batch_size=128 --lr=0.0001 --num_epochs=1 --class_type=entity


5. Classification directly with word Tokenizer and UMLGPT
python uml_classification.py --tokenizer=word --classification_model=uml-gpt --num_layers=6 --num_heads=8 --embed_dim=256 --batch_size=128 --lr=0.0001 --num_epochs=1 --class_type=super_type
python uml_classification.py --tokenizer=word --classification_model=uml-gpt --num_layers=6 --num_heads=8 --embed_dim=256 --batch_size=128 --lr=0.0001 --num_epochs=1 --class_type=entity


6. Classification directly with PLM
python uml_classification.py --classification_model=bert-base-cased --batch_size=128 --lr=0.00001 --num_epochs=1 --class_type=super_type
python uml_classification.py --classification_model=bert-base-cased --batch_size=128 --lr=0.00001 --num_epochs=1 --class_type=entity


7. Classification with pretrained UMLGPT and PLM Tokenizer
### Tokenizer used here should be same as the tokenizer used for pretraining
python uml_classification.py --classification_model=uml-gpt --from_pretrained=models/pre_uml-gpt_tok=bert-base-cased/best_model.pt --tokenizer=bert-base-cased --batch_size=128 --lr=0.0001 --num_epochs=1 --class_type=super_type
python uml_classification.py --classification_model=uml-gpt --from_pretrained=models/pre_uml-gpt_tok=bert-base-cased/best_model.pt --tokenizer=bert-base-cased --batch_size=128 --lr=0.0001 --num_epochs=1 --class_type=entity


8. Classification with pretrained UMLGPT with word Tokenizer
### Tokenizer used here should be same as the tokenizer used for pretraining
python uml_classification.py --classification_model=uml-gpt --from_pretrained=models/pre_uml-gpt_tok=word/best_model.pt --tokenizer=word --batch_size=128 --lr=0.0001 --num_epochs=1 --class_type=super_type

python uml_classification.py --classification_model=uml-gpt --from_pretrained=models/pre_uml-gpt_tok=word/best_model.pt --tokenizer=word --batch_size=128 --lr=0.0001 --num_epochs=1 --class_type=entity


9. Classification with pretrained PLM and PLM Tokenizer
python uml_classification.py --classification_model=uml-gpt --from_pretrained=models/pre_gpt2/best_model --tokenizer=word --batch_size=128 --lr=0.00001 --num_epochs=1 --class_type=super_type
python uml_classification.py --classification_model=uml-gpt --from_pretrained=models/pre_gpt2/best_model --tokenizer=word --batch_size=128 --lr=0.00001 --num_epochs=1 --class_type=entity



13. OntoUML Stereotype Classification
ontouml_classification.py --data_dir=datasets/ontoumlModels --classification_model=uml-gpt --num_layers=6 --num_heads=8 --embed_dim=256 --num_epochs=1


Link Prediction
Using node embeddings from pretrained UMLGPT/LM and GNN-based Link Prediction

10. Link Prediction with pretrained UMLGPT and PLM Tokenizer
python link_prediction.py --from_pretrained=models/pre_uml-gpt_tok=bert-base-cased/best_model.pt --num_layers=2 --embed_dim=256 --tokenizer=bert-base-cased


12. Link Prediction with pretrained PLM and PLM Tokenizer
### Tokenizer used here should be same as the tokenizer used for pretraining
python link_prediction.py --from_pretrained=models/pre_gpt2/best_model --num_layers=2 --embed_dim=256 --tokenizer=gpt2


I have the following files in the project - 


