import streamlit as st
import os
from langchain_community.document_loaders import  PyPDFDirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq

from dotenv import load_dotenv
load_dotenv()

#Load the groq API key
os.environ["GROQ_API_KEY"]=os.getenv("GROQ_API_KEY")
groq_key=os.getenv("GROQ_API_KEY")
os.environ["HF_TOKEN"]=os.getenv("HF_TOKEN")

#Embeddings
embeddings=HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
llm=ChatGroq(groq_api_key=groq_key,model_name="Llama3-8b-8192")

prompt=ChatPromptTemplate.from_template(
    """Answer the question based on the provided context only.
    Please provide the most accurate response based on the question.
    <context>
    {context}
    <context>
    Question:{input}""")

def create_vector_embedding():
    if "vectors" not in st.session_state:
        st.session_state.loader=PyPDFDirectoryLoader("research_papers")
        st.session_state.docs=st.session_state.loader.load()
        st.session_state.text_splitter=RecursiveCharacterTextSplitter(chunk_size=500,chunk_overlap=50)
        st.session_state.final_docs=st.session_state.text_splitter.split_documents(st.session_state.docs[:50])
        st.session_state.embeddings=HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        st.session_state.vectors=FAISS.from_documents(st.session_state.final_docs,st.session_state.embeddings)

st.title("RAG Document Q&A with Groq and llama3")
user_prompt=st.text_input("Enter your query from the research paper")

if st.button("Document Embedding"):
    create_vector_embedding()
    st.write("Vector Database is ready")

import time

if user_prompt:
    document_chain=create_stuff_documents_chain(llm,prompt)
    retriever=st.session_state.vectors.as_retriever()
    retriever_chain=create_retrieval_chain(retriever,document_chain)

    start=time.process_time()
    response=retriever_chain.invoke({'input': user_prompt})
    st.write(f"Response time:{time.process_time()-start}")

    st.write(response['answer'])

    #with a streamlit expander
    with st.expander("Document Similarity search"):
        for i,doc in enumerate(response['context']):
            st.write(doc.page_content)
            st.write('-----------------')