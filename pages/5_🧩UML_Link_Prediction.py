from streamlit.runtime.uploaded_file_manager import UploadedFile
import os
import json
import zipfile
import streamlit as st
from link_prediction import link_prediction
from parameters import parse_args
from constants import UMLGPTMODEL


def graphs_file_validator():
    return graphs_file is not None


def run_validator():
    process_model_zip()
    return \
        graphs_file_validator()


def process_model_zip():
    if isinstance(model_zip, UploadedFile):
        with zipfile.ZipFile(model_zip, 'r') as zip_ref:
            zip_ref.extractall()
            all_files = list()
            for f, _, files in os.walk(model_zip.name.split('.')[0]):
                for file in files:
                    all_files.append(os.path.join(f, file))

            config_file = [f for f in all_files if f.endswith('config.json')][0]
            config = json.load(open(config_file, 'r'))
            model_pretrained_file = [f for f in all_files if f.endswith('best_model.pth') or f.endswith('best_model.pt')][0]
            tokenizer = [f for f in all_files if f.endswith('.pkl') or f.endswith('.pickle')][0]
            if not tokenizer.endswith('.pkl'):
                tokenizer = config['tokenizer']

            args.from_pretrained = os.path.join('models', model_pretrained_file)
            args.tokenizer_file = tokenizer
            args.tokenizer = 'word' if tokenizer.endswith('.pkl') else tokenizer



PAGE_TITLE = "UML Link Prediction"

args = parse_args()
st.set_page_config(page_title=PAGE_TITLE, page_icon="🧩")

st.markdown(PAGE_TITLE)


args.gpt_model = st.selectbox('Node Embedding Model', ['bert-base-cased', UMLGPTMODEL, 'gpt2'])

if args.gpt_model.strip() == UMLGPTMODEL:
    model_zip = st.file_uploader("Pretrained UML GPT Model", type=['zip'])
else:
    finetuned = st.toggle('Use Finetuned Model?', value=False)
    if finetuned:
        model_zip = st.file_uploader("Finetuned HF Model", type=['zip'])
    else:
        model_zip = st.selectbox('Pretrained HF Model', ['bert-base-cased', 'gpt2'])


c1, c2 = st.columns(2)
with c1:
    args.embed_dim = int(st.slider('Embedding Dimension', min_value=128, max_value=1024, value=512, step=128))
    
with c2:
    args.num_layers = int(st.slider('Number of Layers', min_value=1, max_value=12, value=6, step=1))
    

graphs_file = st.file_uploader("Graph Pickle File", type=['pkl', 'gpickle', 'pickle'])
args.graphs_file = graphs_file

start_lp = st.button('Start Link Prediction', on_click=run_validator)
args.stage = 'lp'
if start_lp:
    link_prediction(args)