import sys
__import__('pysqlite3')
import pysqlite3
sys.modules['sqlite3'] = sys.modules["pysqlite3"]
import json
from pydantic import BaseModel
from typing import Dict
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_chroma import Chroma
import streamlit as st
import os


class Document(BaseModel):
    page_content: str
    metadata: Dict


@st.cache_resource
def load_embeddings_model():
    return HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")


@st.cache_resource
def load_documents():
    return json.load(open("data.json", "r"))


@st.cache_resource
def load_embeddings():
    embeddings_model = load_embeddings_model()
    data = load_documents()
    docs = [
        Document(page_content=str(doc), metadata={
                 'source': '/content/data.json', 'seq_num': index+1})
        for index, doc in enumerate(data)
    ]
    return embeddings_model, docs


def create_embeddings():
    persist_directory = './doctors_db'
    embeddings_model, docs = load_embeddings()

    # Check if the vector store already exists on disk
    if os.path.isdir(os.getcwd() + '/doctors_db'):
        st.session_state.db = Chroma(persist_directory=persist_directory,
                                     embedding_function=embeddings_model)
    else:
        st.session_state.db = Chroma.from_documents(
            documents=docs,
            embedding=embeddings_model,
            persist_directory=persist_directory
        )

    return st.session_state.db.as_retriever(search_type="similarity", search_kwargs={"k": 5})

# def main():
#     create_embeddings()

# if __name__ == "__main__":
#     main()
