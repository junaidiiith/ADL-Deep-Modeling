from streamlit.runtime.uploaded_file_manager import UploadedFile
import zipfile
import streamlit as st
from sympy import limit
from uml_classification import uml_classification
from parameters import parse_args
import json
import os
from constants import UMLGPTMODEL
from transformers import \
    (
        AutoModelForMaskedLM, 
        AutoModelForCausalLM, 
        AutoModelForSequenceClassification,
        AutoTokenizer, AutoConfig
    )


def validate():
    process_model_zip()
    if graph_file is None:
        st.error("Please upload a graph file")

    return True
        


def process_model_zip():
    if classification_model == UMLGPTMODEL and isinstance(model_zip, UploadedFile):
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



args = parse_args()
st.set_page_config(page_title="UML Classification", page_icon="🧩")

st.markdown("""UML Class and Supertype Prediction""")

classification_model = st.selectbox('Classification Model', ['bert-base-cased', 'uml-gpt', 'gpt2'])
args.classification_model = classification_model


if classification_model == 'uml-gpt':
    pretrained = st.toggle('Use Pretrained Model?', value=False)
    if pretrained:
        model_zip = st.file_uploader("Pretrained UML GPT Model", type=['zip'])
    else:
        tokenizer = st.selectbox('Tokenizer', ['word', 'bert-base-cased', 'gpt2'])
        args.tokenizer = tokenizer
        args.tokenizer_file = tokenizer
        args.from_pretrained = None
else:
    model_zip = st.file_uploader("Pretrained HF Model", type=['zip'])
    args.classification_model = model_zip.name.split('.')[0]
    args.tokenizer = args.classification_model


classification_type = st.selectbox('Classification Type', ['entity', 'super_type'])
args.class_type = classification_type


if classification_model == UMLGPTMODEL:
    st.markdown("Training Parameters")
    c1, c2, c3 = st.columns(3)
    with c1:
        batch_size = st.slider('Batch Size', min_value=16, max_value=128, value=32, step=16)
        args.batch_size = batch_size

    with c2:
        num_epochs = st.text_input('Number of Epochs', value='10')
        args.num_epochs = int(num_epochs)

    with c3:
        lr = st.text_input('Learning Rate', value='1e-3')
        args.lr = float(lr)


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
graph_file = st.file_uploader("Graph Pickle File", type=['pkl', 'gpickle', 'pickle'])
args.stage = 'cls'
args.graphs_file = graph_file

classification_button = st.button('Start Classification Training', on_click=validate)

if classification_button:
    uml_classification(args)
    st.balloons()
            
# hf_model = st.text_input("Pretrained HF Model")
# if hf_model is not None and 'bert' in hf_model:
#     model = AutoModelForMaskedLM.from_pretrained(hf_model)
#     tokenizer = AutoTokenizer.from_pretrained(hf_model)
    