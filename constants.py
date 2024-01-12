MODELS_DIR = 'models'


EPOCH = 'Epoch'
SPLIT_TYPE = 'split_type'
TRAIN_LOSS = 'Train Loss'
TRAIN_ACC = 'Train Accuracy'
TEST_LOSS = 'Test Loss'
TEST_ACC = 'Test Accuracy'
TRAIN_LABEL = 'train'
TEST_LABEL = 'test'
UNSEEN_LABEL = 'unseen'

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

PRETRAINING = 'pre'
ONTOML_CLS = 'ontouml_cls'
CLASSIFICATION = 'cls'
LINK_PREDICTION = 'lp'

CLASSIFICATION_MODEL = 'classification_model'
GPT_MODEL = 'gpt_model'
EMBEDDING_MODEL = 'embedding_model'
CLASSIFICATION_TYPE = 'class_type'
FROM_PRETRAINED = 'from_pretrained'


uml_plm_names = {
    'BERT Cased': 'bert-base-cased',
    'BERT Uncased': 'bert-base-uncased',
    'GPT2': 'gpt2',
    'UMLGPT': UMLGPTMODEL,
}

tokenizer_names = {
    'Custom Word Tokenizer': 'word',
    'BERT Cased': 'bert-base-cased',
    'BERT Uncased': 'bert-base-uncased',
    'GPT2': 'gpt2',
}

stereotype_classification_model_names = {
    'BERT Cased': 'bert-base-cased',
    'BERT Uncased': 'bert-base-uncased',
    'GPT2': 'gpt2',
    'Conv-BERT': 'YituTech/conv-bert-base',
    'Distilled BERT': 'distilbert-base-uncased-finetuned-sst-2-english'
}

gpt_model_names = {
    'GPT2': 'gpt2',
    'UMLGPT': UMLGPTMODEL,
}


all_plms = {
    'UMLGPT': UMLGPTMODEL,
    'BERT Cased': 'bert-base-cased',
    'BERT Uncased': 'bert-base-uncased',
    'GPT2': 'gpt2',
    'Conv-BERT': 'YituTech/conv-bert-base',
    'Distilled BERT': 'distilbert-base-uncased-finetuned-sst-2-english'
}
