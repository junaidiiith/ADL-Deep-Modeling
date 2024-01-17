import json
import os
import shutil
import streamlit as st
from constants import ONTOML_CLS, TRAINING_CONFIG_JSON, TRAINING_PHASE, phase_mapping
from ontouml_classification import main as ontouml_classification
from parameters import parse_args
from constants import stereotype_classification_model_names as model_names
from pages_input_processing import unzip_models


def validate():
    if args.graphs_file is None:
        st.error("Please upload a graph file")

    return True


args = parse_args()
st.set_page_config(page_title="OntoUML Stereotype Classification", page_icon="🧩")
args.models_dir = os.path.join(args.models_dir, ONTOML_CLS)
args.log_dir = os.path.join(args.log_dir, ONTOML_CLS)
args.inference_models_dir = os.path.join(args.inference_models_dir, ONTOML_CLS)


st.markdown("""## OntoUML Stereotype Classification""")

args.phase = phase_mapping[st.radio('Execution Phase', options=list(phase_mapping.keys()))]


exclude_limit_use_frequent = st.toggle('Use Frequent Classes?', value=False)
if exclude_limit_use_frequent:
    exclude_limit = -1
else:
    exclude_limit = st.slider('Exclude Limit', min_value=100, max_value=1000, step=100, value=100)

c1, c2 = st.columns(2)
with c1:
    distance = st.selectbox('K-hop', range(1, 4))

with c2:
    ontouml_mask_prob = st.slider('OntoUML Mask Probability', min_value=0.1, max_value=0.6, step=0.1, value=0.2)


if args.phase == TRAINING_PHASE:
    args.from_pretrained = model_names[st.selectbox('Classification Model', list(model_names.keys()))]
    args.tokenizer = args.from_pretrained
else:
    trained_onto_cls_models_dir = os.path.join(args.inference_models_dir)
    trained_models = [os.path.join(trained_onto_cls_models_dir, i) for i in os.listdir(trained_onto_cls_models_dir)]
    paths = [i for i in os.listdir(trained_onto_cls_models_dir)]
    args.from_pretrained = os.path.join(trained_onto_cls_models_dir, st.selectbox('Classification Model', paths))
    conf_fp = os.path.join(args.from_pretrained, TRAINING_CONFIG_JSON)
    args.tokenizer = json.load(open(conf_fp, 'r'))['tokenizer']
    


if args.phase == TRAINING_PHASE:
    st.markdown("Training Parameters")
    args.lr = float(st.text_input('Learning Rate', value=2e-5))
    args.num_epochs = st.number_input('Number of Epochs', value=10, min_value=1, max_value=20)


args.batch_size = st.slider('Batch Size', min_value=16, max_value=128, value=32, step=16)
models_file = st.file_uploader("Upload OntoUML Models", type=['zip', 'json'])

args.distance = distance
args.exclude_limit = 100 
args.ontouml_mask_prob = ontouml_mask_prob


start_stereotyping_button = st.button(
    f'{"Start Stereotype Classification Training" if args.phase else "Start Stereotype Classification Inference"}'
)

if start_stereotyping_button:
    args.stage = ONTOML_CLS
    unzip_models(models_file, 'json', args)
    print(models_file)
    validate()
    ontouml_classification(args)

    st.balloons()

    shutil.rmtree(args.graphs_file)

