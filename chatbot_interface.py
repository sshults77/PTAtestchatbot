
import streamlit as st
from langchain.llms import OpenAI
from langchain.chains import RetrievalQA
from langchain.vectorstores import FAISS
from langchain.embeddings.openai import OpenAIEmbeddings
import pickle
import os

st.title("PTA Tutor Chatbot")
st.write("Upload your course materials and ask questions.")

# Load FAISS index
def load_vectorstore():
    with open("vectorstore.pkl", "rb") as f:
        return pickle.load(f)

# Initialize chatbot
vectorstore = load_vectorstore()
qa = RetrievalQA.from_chain_type(llm=OpenAI(), retriever=vectorstore.as_retriever())

# User input
query = st.text_input("Ask a question about your materials:")
if query:
    result = qa.run(query)
    st.write("Answer:", result)
