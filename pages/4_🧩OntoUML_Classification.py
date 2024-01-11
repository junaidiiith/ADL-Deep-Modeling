import streamlit as st
from ontouml_classification import ontouml_classification
from parameters import parse_args

args = parse_args()
st.set_page_config(page_title="OntoUML Stereotype Classification", page_icon="🧩")

st.markdown("""OntoUML Stereotype Classification""")

exclude_limit = st.slider('Exclude Limit', min_value=100, max_value=1000, step=100)
distance = st.slider('K-hop', min_value=1, max_value=3, value=1, step=1)
ontouml_mask_prob = st.slider('OntoUML Mask Probability', min_value=0.1, max_value=0.6, step=0.1)
num_epochs = st.text_input('Number of Epochs', value='10')
data_dir = st.file_uploader("Upload OntoUML Models", type=['zip'])
args.distance = distance
args.exclude_limit = exclude_limit
args.num_epochs = int(num_epochs)
args.ontouml_mask_prob = ontouml_mask_prob
args.data_dir = data_dir
args.tokenizer = args.classification_model

if data_dir is not None:
    ontouml_classification(args)