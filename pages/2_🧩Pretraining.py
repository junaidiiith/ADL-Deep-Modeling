import streamlit as st
from pretraining import main as pretrainer
from parameters import parse_args

def validate():
    if args.graphs_file is None:
        st.error("Graphs file is required")
        return False


    return True


args = parse_args()
st.set_page_config(page_title="Pretraining GPT", page_icon="🧩")

st.markdown("""Pretraining GPT Training Parameters""")

st.markdown("Trainer Parameters")

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


st.markdown("GPT Model Parameters")
args.gpt_model = st.selectbox('GPT Model', ['uml-gpt', 'gpt2'])

if args.gpt_model == 'uml-gpt':

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

    c1, c2, c3 = st.columns(3)
    with c1:
        block_size = st.slider('Block Size', min_value=128, max_value=1024, value=512, step=128)
        args.block_size = int(block_size)
    with c2:
        pooling = st.selectbox('Pooling', ['mean', 'max', 'cls', 'sum', 'last'])
        args.pooling = pooling
    with c3:
        tokenizer = st.selectbox('Tokenizer', ['word', 'bert-base-cased'])
        args.tokenizer = tokenizer

elif args.gpt_model == 'gpt2':
    args.tokenizer = args.gpt_model


# Example file upload
graph_file = st.file_uploader("Graph Pickle File", type=['pkl', 'gpickle', 'pickle'])
args.stage = 'pre'
if graph_file is not None:
    args.graphs_file = graph_file.name

start_button = st.button('Start Pretraining', on_click=validate)
if start_button:
    pretrainer(args)