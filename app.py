import streamlit as st
from langchain_community.document_loaders import PyPDFDirectoryLoader
from PyPDF2 import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
import pandas as pd
import tempfile
import os
import pickle
from langchain_google_genai import GoogleGenerativeAIEmbeddings
st.title('Document QnA with RAG 💡')

st.info('''Welcome to DocuQ! 📄✨

Description: Easily upload your documents and get instant answers to your questions. DocuQ provides quick summaries, accurate Q&A, and interactive quizzes to help you understand your content better. Enjoy a seamless and secure experience with our user-friendly interface.''')


file=st.file_uploader(label='Upload your Pdf Here',type='pdf')
if file is not None:
    pdf_reader=PdfReader(file)
    st.write(pdf_reader)
    text=''
    for pages in pdf_reader.pages:
        text += pages.extract_text()
    st.write(text)
    embeddings=GoogleGenerativeAIEmbeddings()
    splitter=RecursiveCharacterTextSplitter(chunk_size=100,chunk_overlap=20)
    splitted_text=splitter.split_text(text=text)
    st.write(splitted_text)








user_prompt=st.chat_input('Enter your Query realted to your Document')
if user_prompt:
    st.chat_message('user').markdown(user_prompt)