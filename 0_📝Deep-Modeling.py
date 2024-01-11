import streamlit as st

def main():
    st.title('Graph Language Modeling for Conceptual Modeling Languages')
    contents = open("README.md", "r").read()
    st.markdown(contents)

if __name__ == '__main__':
    main()
