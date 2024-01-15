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
st.set_page_config(page_title="UML Classification", page_icon="🧩")

st.markdown("""## UML Class and Supertype Prediction""")

args.phase = phase_mapping[st.radio('Execution Phase', options=list(phase_mapping.keys()))]

# if args.phase == INFERENCE_PHASE:
#     available_classification_models = [GPT2_LABEL, UMLGPT_LABEL]
# else:
available_classification_models = list(uml_plm_names.keys())

classification_model = uml_plm_names[st.selectbox('Classification Model', available_classification_models)]
args.classification_model = classification_model


if classification_model == UMLGPTMODEL:
    if args.phase == TRAINING_PHASE:
        pretrained = st.toggle('Use Pretrained Model?', value=False)
    else:
        pretrained = True

    if pretrained:
        plms = get_plms(args.models_dir, PRETRAINING, classification_model)
        plm_dir = st.selectbox('Pretrained Model', plms)
        process_uml_gpt_dir(plm_dir)
    else:
        tokenizer = tokenizer_names[st.selectbox('Tokenizer', list(tokenizer_names.keys()))]
        args.tokenizer = tokenizer

else:
    if args.phase == TRAINING_PHASE:
        fine_tuned = st.toggle('Use Fine Tuned Model?', value=False)
    else:
        fine_tuned = True

    if fine_tuned:
        # if not classification_model in [UMLGPTMODEL, 'gpt2']:
        #     st.error("Fine Tuned Model is only available for UMLGPT and gpt2")

        plms = get_plms(args.models_dir, PRETRAINING, classification_model)
        plms += get_plms(args.models_dir, UML_CLASSIFICATION, classification_model)
        if len(plms) == 0:
            st.error("No Fine Tuned Model Available for this model")

        plm_dir = os.path.join(args.models_dir, st.selectbox('Pretrained HF Model', plms))
        args.from_pretrained = plm_dir
    else:
        args.from_pretrained = args.classification_model


classification_type = st.selectbox('Classification Type', classification_types.keys())
args.class_type = classification_types[classification_type]

args.batch_size = st.slider('Batch Size', min_value=16, max_value=128, value=32, step=16)


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

