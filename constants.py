import torch

UPLOADED_DATA_DIR = 'uploaded_data'

EPOCH = 'Epoch'
SPLIT_TYPE = 'split_type'
TRAIN_LOSS = 'Train Loss'
TRAIN_ACC = 'Train Accuracy'
TEST_LOSS = 'Test Loss'
TEST_ACC = 'Test Accuracy'
TRAIN_LABEL = 'train'
TEST_LABEL = 'test'
UNSEEN_LABEL = 'unseen'
LOSS = 'loss'

EVAL_LOSS = 'eval_loss'


SSP = "<superType>"
ESP = "</superType>"
SEN = "<entity>"
EEN = "</entity>"

SRP = "<relations>"
ERP = "</relations>"

PAD = "<pad>"
UNK = "<unk>"
SOS = "<s>"
EOS = "</s>"
MASK = "<mask>"
SEP = "<sep>"

SPECIAL_TOKENS = [PAD, UNK, SOS, EOS, MASK, SEP, SSP, ESP, SEN, EEN, SRP, ERP]


classification_types = {
    'Entity Classification': 'entity',
    'Super Type Classification': 'super_type'
}


UMLGPTMODEL = 'uml-gpt'
WORD_TOKENIZER = 'word'
TOKENIZER_FILE = 'tokenizer_file'

PRETRAINING = 'pre'
ONTOML_CLS = 'ontouml_cls'
UML_CLASSIFICATION = 'cls'
LINK_PREDICTION = 'lp'
PRETRAINING_PY = 'pretraining.py'
UML_CLASSIFICATION_PY = 'uml_classification.py'
ONTOML_CLS_PY = 'ontouml_classification.py'
LINK_PREDICTION_PY = 'link_prediction.py'

task2file_map = {
    ONTOML_CLS: ONTOML_CLS_PY,
    UML_CLASSIFICATION: UML_CLASSIFICATION_PY,
    PRETRAINING: PRETRAINING_PY,
    LINK_PREDICTION: LINK_PREDICTION_PY
}
RUN_COMMAND = 'run_command'

DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')



CLASSIFICATION_MODEL = 'classification_model'
GPT_MODEL = 'gpt_model'
EMBEDDING_MODEL = 'embedding_model'
CLASSIFICATION_TYPE = 'class_type'
FROM_PRETRAINED = 'from_pretrained'
TOKENIZER_LABEL = 'tokenizer'
BEST_MODEL_LABEL = 'best_model.pt'
SAFE_TENSORS_LABEL = 'model.safetensors'

CUSTOM_WORD_TOKENIZER = 'Custom Word Tokenizer'

BERT_BASE_CASED_LABEL = 'BERT Base Cased'
BERT_BASE_UNCASED_LABEL = 'BERT Base UnCased'
GPT2_LABEL = 'GPT2'
UMLGPT_LABEL = 'UMLGPT'
CONV_BERT_LABEL = 'Convolution BERT'
DISTILLED_BERT_LABEL = 'Distilled BERT'


uml_plm_names = {
    BERT_BASE_CASED_LABEL: 'bert-base-cased',
    BERT_BASE_UNCASED_LABEL: 'bert-base-uncased',
    GPT2_LABEL: 'gpt2',
    UMLGPT_LABEL: UMLGPTMODEL,
}

tokenizer_names = {
    CUSTOM_WORD_TOKENIZER: 'word',
    BERT_BASE_CASED_LABEL: 'bert-base-cased',
    BERT_BASE_UNCASED_LABEL: 'bert-base-uncased',
    GPT2_LABEL: 'gpt2',
}

stereotype_classification_model_names = {
    BERT_BASE_CASED_LABEL: 'bert-base-cased',
    BERT_BASE_UNCASED_LABEL: 'bert-base-uncased',
    GPT2_LABEL: 'gpt2',
    CONV_BERT_LABEL: 'YituTech/conv-bert-base',
    DISTILLED_BERT_LABEL: 'distilbert-base-uncased-finetuned-sst-2-english'
}

gpt_model_names = {
    GPT2_LABEL: 'gpt2',
    'UMLGPT': UMLGPTMODEL,
}


all_plms = {
    UMLGPT_LABEL: UMLGPTMODEL,
    BERT_BASE_CASED_LABEL: 'bert-base-cased',
    BERT_BASE_UNCASED_LABEL: 'bert-base-uncased',
    GPT2_LABEL: 'gpt2',
    CONV_BERT_LABEL: 'YituTech/conv-bert-base',
    DISTILLED_BERT_LABEL: 'distilbert-base-uncased-finetuned-sst-2-english'
}


TRAINING_PHASE = 'training'
TRAINING_PHASE_LABEL = 'Train Model'

INFERENCE_PHASE = 'inference'
INFERENCE_PHASE_LABEL = 'Run Inference'

phase_mapping = {
    TRAINING_PHASE_LABEL: TRAINING_PHASE,
    INFERENCE_PHASE_LABEL: INFERENCE_PHASE
}

TRAINING_CONFIG_JSON = 'training_config.json'