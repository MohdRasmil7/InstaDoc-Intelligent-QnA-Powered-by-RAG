import streamlit as st
from langchain_community.document_loaders import PyPDFDirectoryLoader
from PyPDF2 import PdfReader
import pandas as pd
import tempfile
import os
import pickle
st.title('Document QnA with RAG 💡')

st.info('''Welcome to DocuQ! 📄✨

Description: Easily upload your documents and get instant answers to your questions. DocuQ provides quick summaries, accurate Q&A, and interactive quizzes to help you understand your content better. Enjoy a seamless and secure experience with our user-friendly interface.''')


file=st.file_uploader(label='Upload your Pdf Here',type='pdf')
if file is not None:
    pdf_reader=PdfReader(file)
    st.write(pdf_reader)

st.session_state.loader=PyPDFDirectoryLoader('/','.pkl')
#store_name=file.name[:-4]
#with open(f'{store_name}.pkl','wb') as f:
#            pickle.dump(store_name,f)






user_prompt=st.chat_input('Enter your Query realted to your Document')
if user_prompt:
    st.chat_message('user').markdown(user_prompt)