import json
import os
import shutil
import streamlit as st
from constants import PRETRAINING, \
    TRAINING_PHASE, UMLGPTMODEL, WORD_TOKENIZER, phase_mapping
from pretraining import main as pretrainer
from parameters import parse_args
from constants import gpt_model_names, tokenizer_names
from pages_input_processing import unzip_models, get_plms


def validate():
    if graph_file is None:
        st.error("Graphs file is required")
        return False


    return True


def process_uml_gpt_dir(uml_plm_dir):
    all_files = [f for f in os.listdir(os.path.join(args.models_dir, uml_plm_dir)) if not f.startswith('.')]
    config_file = [f for f in all_files if f.endswith('config.json')][0]
    conf_file_path = os.path.join(args.models_dir, uml_plm_dir, config_file)
    config = json.load(open(conf_file_path, 'r'))

    model_pretrained_file = [os.path.join(uml_plm_dir, f) for f in all_files if f.endswith('best_model.pth') or f.endswith('best_model.pt')][0]
    tokenizer = [os.path.join(uml_plm_dir, f) for f in all_files if f.endswith('.pkl') or f.endswith('.pickle')]
    if len(tokenizer) and tokenizer[0].endswith('.pkl'):
        tokenizer = tokenizer[0]
        args.tokenizer_file = os.path.join(args.models_dir, tokenizer)
    else:
        tokenizer = config['tokenizer']

    args.from_pretrained = os.path.join(args.models_dir, model_pretrained_file)
    args.tokenizer = WORD_TOKENIZER if tokenizer.endswith('.pkl') else tokenizer
    


args = parse_args()
st.set_page_config(page_title="Pretraining GPT", page_icon="🧩")

st.markdown("## Train Generative Models for UML models")

args.phase = phase_mapping[st.radio('Execution Phase', options=list(phase_mapping.keys()))]
 

if args.phase == TRAINING_PHASE:
    st.markdown("""Pretraining GPT Training Parameters""")
    st.markdown("Trainer Parameters")

    c2, c3 = st.columns(2)

    with c2:
        num_epochs = st.text_input('Number of Epochs', value='10')
        args.num_epochs = int(num_epochs)

    with c3:
        lr = st.text_input('Learning Rate', value='1e-3')
        args.lr = float(lr)


st.markdown("GPT Model Parameters")
args.gpt_model = gpt_model_names[st.selectbox('GPT Model', list(gpt_model_names.keys()))]

args.batch_size = st.slider('Batch Size', min_value=16, max_value=128, value=32, step=16)


if args.gpt_model == UMLGPTMODEL:
    if args.phase == TRAINING_PHASE:
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
    
        tokenizer = tokenizer_names[st.selectbox('Tokenizer', list(tokenizer_names.keys()))]
        args.tokenizer = tokenizer
    else:
        plms = get_plms(args.models_dir, PRETRAINING, args.gpt_model)
        plm_dir = st.selectbox('Pretrained Model', plms)
        process_uml_gpt_dir(plm_dir)

else:
    if args.phase == TRAINING_PHASE:
        args.tokenizer = args.gpt_model
    else:
        plms = get_plms(args.models_dir, PRETRAINING, args.gpt_model)
        plm_dir = os.path.join(args.models_dir, st.selectbox('Pretrained HF Model', plms))
        args.from_pretrained = plm_dir


# Example file upload
graph_file = st.file_uploader("Upload ECore Models", type=['zip'])
args.stage = PRETRAINING

start_button = st.button(
    f'{"Start Pretraining" if args.phase == TRAINING_PHASE else "Run Inference"}', on_click=validate
)
if start_button:
    unzip_models(graph_file, 'ecore', args)
    pretrainer(args)
    st.balloons()

    shutil.rmtree(args.graphs_file)
