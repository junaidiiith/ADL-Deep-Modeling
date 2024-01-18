import shutil
from pages_input_processing import unzip_models
import os
import json
from pages_input_processing import get_plms
import streamlit as st
from link_prediction import main as link_prediction
from parameters import parse_args
from constants import *


def graphs_file_validator():
    return graphs_file is not None


def run_validator():
    return \
        graphs_file_validator()


def process_uml_gpt_dir(uml_plm_dir):
    all_files = [f for f in os.listdir(uml_plm_dir) if not f.startswith('.')]
    config_file = [f for f in all_files if f.endswith(TRAINING_CONFIG_JSON)][0]
    conf_file_path = os.path.join(uml_plm_dir, config_file)
    config = json.load(open(conf_file_path, 'r'))

    model_pretrained_file = [os.path.join(uml_plm_dir, f) for f in all_files if f.endswith('best_model.pth') or f.endswith('best_model.pt')][0]

    tokenizer = [os.path.join(uml_plm_dir, f) for f in all_files if f.endswith('.pkl') or f.endswith('.pickle')]
    if len(tokenizer) and tokenizer[0].endswith('.pkl'):
        tokenizer = tokenizer[0]
        args.tokenizer_file = os.path.join(args.inference_models_dir, tokenizer)
    else:
        tokenizer = config['tokenizer']

    args.from_pretrained = model_pretrained_file
    args.tokenizer = WORD_TOKENIZER if tokenizer.endswith('.pkl') else tokenizer
    

PAGE_TITLE = "## UML Link Prediction"

args = parse_args()
st.set_page_config(page_title=PAGE_TITLE, page_icon="🧩")
plms_repo = f'{args.inference_models_dir}/{PRETRAINING}'
parent_dir = args.inference_models_dir

args.models_dir = f'{args.models_dir}/{LINK_PREDICTION}'
args.inference_models_dir = f'{args.inference_models_dir}/{LINK_PREDICTION}'
st.markdown(PAGE_TITLE)


args.phase = phase_mapping[st.radio('Execution Phase', options=list(phase_mapping.keys()))]
args.embedding_model = all_plms[st.selectbox('Node Embedding Model', list(all_plms.keys()))]

gnns_location = os.path.join(parent_dir, LINK_PREDICTION, args.embedding_model)

if args.embedding_model.strip() == UMLGPTMODEL:
    plms = [os.path.join(plms_repo, i) for i in get_plms(plms_repo, PRETRAINING, args.embedding_model)]
    plm_dir = st.selectbox('Pretrained Model', plms)
    process_uml_gpt_dir(plm_dir)
else:
    finetuned = st.toggle('Use Finetuned Model?', value=False)
    if finetuned:
        print(args.inference_models_dir, args.embedding_model)
        plms = [os.path.join(plms_repo, plm) for plm in get_plms(plms_repo, PRETRAINING, args.embedding_model)]
        
        if len(plms) == 0:
            st.error(f'No finetuned models found for {args.embedding_model}!\nSelect a different PLM or train a new {args.embedding_model} model')
            st.stop()
        # plm_dir = os.path.join(args.models_dir, st.selectbox('Pretrained HF Model', all_plms_possible))
        plm_dir = st.selectbox('Finetuned HF Model', plms)
        args.embedding_model = plm_dir
        args.tokenizer = plm_dir
        
    else:
        args.tokenizer = args.embedding_model

if args.phase == TRAINING_PHASE:
    st.markdown("Model Parameters")
    c1, c2, c3 = st.columns(3)
    with c1:
        args.embed_dim = int(st.slider('Embedding Dimension', min_value=128, max_value=1024, value=512, step=128))
        
    with c2:
        args.num_layers = int(st.slider('Number of Layers', min_value=1, max_value=12, value=6, step=1))

    with c3:
        args.num_heads = int(st.slider('Number of Heads', min_value=1, max_value=12, value=8, step=1))

        

    st.markdown("Optimizer Parameters")
    c1, c2, c3 = st.columns(3)
    with c1:
        args.lr = float(st.text_input('Learning Rate', value='1e-3'))

    with c2:
        args.weight_decay = float(st.text_input('Weight Decay', value='1e-4'))

    with c3:
        args.warmup_steps = int(st.text_input('Warmup Steps', value='100'))

else:
    trained_gnns = [
        os.path.join(gnns_location, i) for i in os.listdir(f"{gnns_location}") if os.path.isdir(os.path.join(gnns_location, i))]
    pretrained_gnn = st.selectbox('Pretrained GNN', trained_gnns)
    args.gnn_location = pretrained_gnn


st.markdown("Training Parameters")
c1, c2, c3 = st.columns(3)

with c1:
    args.batch_size = st.slider('Graph Batch Size', min_value=16, max_value=128, value=32, step=16)
with c2:
    args.num_epochs = int(st.text_input('Number of Epochs', value=1))
with c3:
    args.test_size = float(st.text_input('Test Size', value=0.2))

args.block_size = 512

graphs_file = st.file_uploader("Upload Ecore Models", type=['zip'])


start_lp = st.button('Start Link Prediction Training', on_click=run_validator)
args.stage = LINK_PREDICTION
if start_lp:
    unzip_models(graphs_file, 'ecore', args)
    link_prediction(args)
    st.balloons()

    shutil.rmtree(args.graphs_file)