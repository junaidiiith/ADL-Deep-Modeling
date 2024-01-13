import streamlit as st
from constants import ONTOML_CLS
from ontouml_classification import main as ontouml_classification
from parameters import parse_args

def validate():
    if args.data_dir is None:
        st.error("Please upload a graph file")

    return True

model_names = {
    'BERT Cased': 'bert-base-cased',
    'BERT Uncased': 'bert-base-uncased',
    'GPT2': 'gpt2',
    'Conv-BERT': 'YituTech/conv-bert-base',
    'Distilled BERT': 'distilbert-base-uncased-finetuned-sst-2-english'
}

args = parse_args()
st.set_page_config(page_title="OntoUML Stereotype Classification", page_icon="🧩")

st.markdown("""OntoUML Stereotype Classification""")



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

st.markdown("Training Parameters")
model_name = st.selectbox('Classification Model', list(model_names.keys()))

if model_name:
    args.from_pretrained = model_names[model_name]

c1, c2 = st.columns(2)
with c1:
    args.num_epochs = st.number_input('Number of Epochs', value=10, min_value=1, max_value=20)
with c2:
    args.batch_size = st.slider('Batch Size', min_value=16, max_value=128, value=32, step=16)


data_dir = st.file_uploader("Upload OntoUML Models", type=['zip'])

args.distance = distance
args.exclude_limit = exclude_limit
args.ontouml_mask_prob = ontouml_mask_prob
args.data_dir = data_dir


start_stereotyping_button = st.button('Start Stereotype Classification Training', on_click=validate)
if start_stereotyping_button:
    args.exclude_limit = 1
    args.stage = ONTOML_CLS
    ontouml_classification(args)
    st.balloons()