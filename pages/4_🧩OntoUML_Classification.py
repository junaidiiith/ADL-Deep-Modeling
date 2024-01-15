import os
import shutil
import streamlit as st
from constants import ONTOML_CLS, TRAINING_PHASE, phase_mapping
from ontouml_classification import main as ontouml_classification
from parameters import parse_args
from constants import stereotype_classification_model_names as model_names
from pages_input_processing import unzip_ontouml_models

STEREOTYPE_CLASSIFICATION_REPO = 'models/stereotype_classification'

def validate():
    if args.data_dir is None:
        st.error("Please upload a graph file")

    return True


args = parse_args()
st.set_page_config(page_title="OntoUML Stereotype Classification", page_icon="🧩")

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
    model_name = st.selectbox('Classification Model', list(model_names.keys()))

    if model_name:
        args.from_pretrained = model_names[model_name]

else:
    model_name = st.selectbox('Classification Model', list(os.listdir(STEREOTYPE_CLASSIFICATION_REPO)))

    if model_name:
        args.from_pretrained = os.path.join(STEREOTYPE_CLASSIFICATION_REPO, model_name)

if args.phase == TRAINING_PHASE:
    st.markdown("Training Parameters")
    args.num_epochs = st.number_input('Number of Epochs', value=10, min_value=1, max_value=20)

args.batch_size = st.slider('Batch Size', min_value=16, max_value=128, value=32, step=16)


data_zip_file = st.file_uploader("Upload OntoUML Models", type=['zip'])

args.distance = distance
args.exclude_limit = exclude_limit
args.ontouml_mask_prob = ontouml_mask_prob


start_stereotyping_button = st.button(
    f'{"Start Stereotype Classification Training" if args.phase else "Start Stereotype Classification Inference"}', on_click=validate)

if start_stereotyping_button:
    args.stage = ONTOML_CLS
    unzip_ontouml_models(data_zip_file, args)
    ontouml_classification(args)

    shutil.rmtree(args.data_dir)
    st.balloons()