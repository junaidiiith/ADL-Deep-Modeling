import shutil
import streamlit as st
from uml_classification import main as uml_classification
from parameters import parse_args
import json
import os
from constants import *

from pages_input_processing import unzip_models, get_plms


def validate():
    if models_file is None:
        st.error("Please upload a graph file")

    return True


def process_uml_gpt_dir(uml_plm_dir):
    

    if WORD_TOKENIZER in uml_plm_dir:
        args.tokenizer = WORD_TOKENIZER
        return

    all_files = [f for f in os.listdir(uml_plm_dir) if not f.startswith('.')]
    config_file = [f for f in all_files if f == TRAINING_CONFIG_JSON][0]
    conf_file_path = os.path.join(uml_plm_dir, config_file)
    config = json.load(open(conf_file_path, 'r'))
    # print(config_file, conf_file_path, config)
    model_pretrained_file = [os.path.join(uml_plm_dir, f) for f in all_files if f.endswith('best_model.pth') or f.endswith('best_model.pt')][0]
    tokenizer = [os.path.join(uml_plm_dir, f) for f in all_files if f.endswith('.pkl') or f.endswith('.pickle')]
    if len(tokenizer) and tokenizer[0].endswith('.pkl'):
        tokenizer = tokenizer[0]
        args.tokenizer_file = tokenizer
    else:
        tokenizer = config['tokenizer']
    # print("Model Pretrained File", model_pretrained_file)
    args.from_pretrained = model_pretrained_file
    args.tokenizer = WORD_TOKENIZER if tokenizer.endswith('.pkl') else tokenizer


def get_all_plms(model, dir):
    parent = os.path.dirname(dir)

    pre_dir = os.path.join(parent, PRETRAINING)
    pre_plms = get_plms(pre_dir, PRETRAINING, model)
    pre_plm_paths = [os.path.join(pre_dir, i) for i in pre_plms]

    cls_dir = os.path.join(parent, UML_CLASSIFICATION)
    cls_plms = get_plms(cls_dir, UML_CLASSIFICATION, model)
    cls_plm_paths = [os.path.join(cls_dir, i) for i in cls_plms]
    
    all_plms = pre_plm_paths + cls_plm_paths

    return all_plms


def get_model_plm():
    plms = get_all_plms(UMLGPTMODEL, args.inference_models_dir)
    plm_dir = st.selectbox('Pretrained Model', plms)
    return plm_dir


def set_pretrained_hf_config(model):
    plms = get_all_plms(model, args.inference_models_dir)
    if len(plms):
        plm_dir = st.selectbox('Pretrained Model', plms)
    else:
        plm_dir = model
    args.from_pretrained = plm_dir
    args.tokenizer = plm_dir


def get_uml_gpt_tokenizer(dir):
    if 'tok=word' in dir:
        args.tokenizer = WORD_TOKENIZER
        args.tokenizer_file = os.path.join(dir, 'tokenizer.pkl')
    elif 'tok=bert-base-cased' in dir:
        args.tokenizer = 'bert-base-cased'
    elif 'tok=bert-base-uncased' in dir:
        args.tokenizer  = 'bert-base-uncased'
    elif 'gpt2' in dir:
        args.tokenizer = 'gpt2'



args = parse_args()
st.set_page_config(page_title="UML Classification", page_icon="🧩")
args.models_dir = os.path.join(args.models_dir, UML_CLASSIFICATION)
args.log_dir = os.path.join(args.log_dir, UML_CLASSIFICATION)
args.inference_models_dir = os.path.join(args.inference_models_dir, UML_CLASSIFICATION)


st.markdown("""## UML Class and Supertype Prediction""")

args.phase = phase_mapping[st.radio('Execution Phase', options=list(phase_mapping.keys()))]

available_classification_models = list(uml_plm_names.keys())

classification_model = uml_plm_names[st.selectbox('Classification Model', available_classification_models)]
args.classification_model = classification_model


if classification_model == UMLGPTMODEL:
    if args.phase == TRAINING_PHASE:
        pretrained = st.toggle('Use Pretrained Model?', value=False)
        if pretrained:
            plm_path = get_model_plm()
            get_uml_gpt_tokenizer(plm_path)
            args.from_pretrained = os.path.join(plm_path, BEST_MODEL_LABEL)
        else:
            tokenizer = tokenizer_names[st.selectbox('Select Tokenizer', list(tokenizer_names.keys()))]
            args.tokenizer = tokenizer
    else:
        plm_path = get_model_plm()
        get_uml_gpt_tokenizer(plm_path)
        args.from_pretrained = os.path.join(plm_path, BEST_MODEL_LABEL)

else:
    if args.phase == TRAINING_PHASE:
        fine_tuned = st.toggle('Use Fine Tuned Model?', value=False)
        if fine_tuned:
            set_pretrained_hf_config(classification_model)
        else:
            args.from_pretrained = classification_model
    else:
        
        all_plms = get_all_plms(classification_model, args.inference_models_dir) + [classification_model]

        if len(all_plms) == 0:
            st.error("No Fine Tuned Model Available for this model")

        plm_dir = st.selectbox('Pretrained HF Model', all_plms)
        args.from_pretrained = plm_dir
    
    args.tokenizer = classification_model


c1, c2 = st.columns(2)
with c1:
    args.class_type = classification_types[st.selectbox('Classification Type', classification_types.keys())]
with c2:
    args.batch_size = st.slider('Batch Size', min_value=16, max_value=1024, value=32, step=16)


if args.phase == TRAINING_PHASE:
    st.markdown("Training Parameters")
    c2, c3 = st.columns(2)
    
    with c2:
        num_epochs = st.text_input('Number of Epochs', value='10')
        args.num_epochs = int(num_epochs)

    with c3:
        lr = st.text_input('Learning Rate', value='1e-3')
        args.lr = float(lr)

    if classification_model == UMLGPTMODEL and not pretrained:
        st.markdown("Model Parameters")
        if classification_model == 'uml-gpt':
            c1, c2, c3 = st.columns(3)
            with c1:
                embed_dim = st.text_input('Embedding Dimension', value='128')    
                args.embed_dim = int(embed_dim)
            with c2:
                num_layers = st.slider('Number of Layers', min_value=1, max_value=12, value=6, step=1)
                args.num_layers = int(num_layers)
            with c3:
                num_heads = st.slider('Number of Heads', min_value=1, max_value=12, value=8, step=1)
                args.num_heads = int(num_heads)

            c1, c2 = st.columns(2)
            with c1:
                block_size = st.slider('Block Size', min_value=128, max_value=1024, value=512, step=128)
                args.block_size = int(block_size)
            with c2:
                pooling = st.selectbox('Pooling', ['mean', 'max', 'cls', 'sum', 'last'])
                args.pooling = pooling

# Example file upload
models_file = st.file_uploader("Upload ECore Models", type=['zip', 'ecore'])
args.stage = UML_CLASSIFICATION

classification_button = st.button(
    f'{"Start Classification Training" if args.phase == TRAINING_PHASE else "Run Classification Inference"}', on_click=validate)

if classification_button:
    unzip_models(models_file, 'ecore', args)

    uml_classification(args)
    st.balloons()

    shutil.rmtree(args.graphs_file)

