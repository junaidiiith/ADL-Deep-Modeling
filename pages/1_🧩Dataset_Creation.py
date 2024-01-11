import os
import pandas as pd
import streamlit as st
from model2nx import get_graphs_from_zip_file
from model2nx import write_graphs_to_file
from model2nx import clean_graph_set

UPLOAD_DIR = "uploaded_data/"
st.title('Dataset Creation from UML Models')


st.markdown("Graph Dataset Creation")
test_size = st.text_input('Test size', value='0.2')
# seed = st.text_input('Seed', value='42')
graph_file = st.file_uploader("Graph dataset", type=['zip'])

if graph_file is not None:
    graphs_file_path = UPLOAD_DIR + graph_file.name.split(".")[0] + "_graphs.pkl"
    # with st.spinner('Loading graphs...'):

if graph_file is not None and not os.path.exists(graphs_file_path):
    graphs = get_graphs_from_zip_file(graph_file)
    cleaned_graphs = clean_graph_set(graphs)
    write_graphs_to_file(cleaned_graphs, graphs_file_path)

    st.markdown("Total graphs: " + str(len(graphs)))

    num_nodes = [g.number_of_nodes() for g in graphs]
    num_edges = [g.number_of_edges() for g in graphs]
    
    df = pd.DataFrame({'Number of Nodes': num_nodes, 'Number of Edges': num_edges})
    st.scatter_chart(df)

    ### Download the file
    with st.spinner('Downloading graphs...'):
        with open(graphs_file_path, 'rb') as f:
            st.download_button(
                label="Download Graphs Binary File",
                data=f,
                file_name=graph_file.name.split(".")[0] + "_graphs.pkl",
                mime='*/*'
            )

