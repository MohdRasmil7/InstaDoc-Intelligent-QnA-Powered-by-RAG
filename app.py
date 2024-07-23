import streamlit as st
from langchain_community.document_loaders import PyPDFDirectoryLoader
from PyPDF2 import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
import pandas as pd
from langchain_community.vectorstores import FAISS
import os
import pickle
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from dotenv import load_dotenv
load_dotenv()

os.environ['GROQ_API_KEY']=os.getenv('GROQ_API_KEY')
os.environ['GOOGLE_API_KEY']=os.getenv('GOOGLE_API_KEY')

os.environ['GROQ_API_KEY']=os.getenv('GROQ_API_KEY')
os.environ['GOOGLE_API_KEY']=os.getenv('GOOGLE_API_KEY')


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
    
    embeddings=GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    splitter=RecursiveCharacterTextSplitter(chunk_size=100,chunk_overlap=20)
    splitted_text=splitter.split_text(text=text)

    vector_stores=FAISS.from_texts(embedding=embeddings,texts=splitted_text)
    
    store_name=file.name[:-4]
    if os.path.exists(f'{store_name}.pkl'):
        with open(f'{store_name}.pkl','rb') as f:
            vector_stores=pickle.loads(f)
            st.write('embedding loaded already')
    else:
        with open(f'{store_name}.pkl','wb') as f:
            pickle.dump(vector_stores,f)

    
   








user_prompt=st.chat_input('Enter your Query realted to your Document')
if user_prompt:
    st.chat_message('user').markdown(user_prompt)