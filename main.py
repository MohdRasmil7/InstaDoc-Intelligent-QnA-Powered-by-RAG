import streamlit as st
from PyPDF2 import PdfReader
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from nltk.tokenize import sent_tokenize
from langchain_groq import ChatGroq
from langchain.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.retrieval import create_retrieval_chain
from dotenv import load_dotenv
import os
import nltk

# Explicitly download NLTK data
nltk.download('punkt', quiet=True)

load_dotenv()

os.environ['GROQ_API_KEY'] = os.getenv('GROQ_API_KEY')

st.set_page_config(page_title="InstaDoc: Document Q&A powered by RAG")

st.title('InstaDoc: Document Q&A powered by RAG 💡')
st.info('''Welcome to InstaDoc! 📄✨

Description: Upload your documents and instantly get answers to your queries. InstaDoc offers precise Q&A, detailed document analysis, and seamless interaction, all through a user-friendly interface.''')

llm = ChatGroq(model='mixtral-8x7b-32768')

prompt = ChatPromptTemplate.from_template(
    '''
As an expert in document analysis, you have a comprehensive understanding of the content within the uploaded PDFs and are proficient in extracting and interpreting relevant information. Users will pose questions related to the document, seeking accurate and concise answers.

If a question pertains to information not contained within the document, kindly inform the user that the document does not contain the requested information.

In cases where the answer is not clear from the document, it is important to maintain honesty by stating 'I don't know' rather than providing potentially incorrect information.

Below is a snippet of context from the relevant section of the document, which will not be shown to users.
<context>
Context: {context}
Question: {input}
</context>
Your response should consist solely of useful information extracted from the document. Be polite and enhance your answer with detailed information.

Useful information:

'''
)

# Initialize session state
if 'vector_store' not in st.session_state:
    st.session_state.vector_store = None

st.sidebar.header('Upload Your document here')
file = st.sidebar.file_uploader(label='Upload PDF', type='pdf', label_visibility="collapsed")

if file is not None:
    if 'last_uploaded_file' not in st.session_state or st.session_state.last_uploaded_file != file.name:
        st.session_state.last_uploaded_file = file.name
        st.session_state.vector_store = None

    if st.session_state.vector_store is None:
        try:
            pdf_reader = PdfReader(file)
            text = ''
            for page in pdf_reader.pages:
                text += page.extract_text()

            # Initialize HuggingFaceEmbeddings with the correct parameters
            embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

            splitted_text = sent_tokenize(text)  # Using NLTK for text splitting

            st.session_state.vector_store = Chroma.from_texts(texts=splitted_text, embedding=embeddings)  # Using ChromaDB
            st.success('Embedding created and stored in session ✨')
        except Exception as e:
            st.error(f"An error occurred while processing the document: {str(e)}")
    else:
        st.info('Using existing embedding from session 💬')

    user_prompt = st.chat_input('Enter your Query related to your Document')
    if user_prompt:
        try:
            st.chat_message('user').markdown(user_prompt)
            document_chain = create_stuff_documents_chain(llm, prompt)
            retriever = st.session_state.vector_store.as_retriever()
            retriever_chain = create_retrieval_chain(retriever, document_chain)
            response = retriever_chain.invoke({'input': user_prompt})

            st.chat_message('assistant').markdown(response['answer'])
        except Exception as e:
            st.error(f"An error occurred while processing your query: {str(e)}")
else:
    st.warning("Please upload a PDF document to start.")