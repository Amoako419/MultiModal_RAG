import streamlit as st
from langchain.embeddings import GoogleGenerativeAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.llms import GoogleGenerativeAI
from langchain.chains import RetrievalQA
from langchain.document_loaders import TextLoader, PyPDFLoader, Docx2txtLoader
from langchain.text_splitter import CharacterTextSplitter
import os

# Set up API keys (replace with your actual API key)
os.environ["GOOGLE_API_KEY"] = "your_google_api_key"

# Initialize embeddings model
embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

# Load documents from multiple formats
def load_documents():
    documents = []
    txt_loader = TextLoader("data.txt")
    pdf_loader = PyPDFLoader("data.pdf")
    docx_loader = Docx2txtLoader("data.docx")
    documents.extend(txt_loader.load())
    documents.extend(pdf_loader.load())
    documents.extend(docx_loader.load())
    return documents

documents = load_documents()

# Split text into chunks
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
texts = text_splitter.split_documents(documents)

# Create FAISS vector store
vector_store = FAISS.from_documents(texts, embeddings)
retriever = vector_store.as_retriever()

# Initialize Gemini LLM
llm = GoogleGenerativeAI(model="gemini-pro")

# Create RAG pipeline
qa_chain = RetrievalQA(llm=llm, retriever=retriever)

# Streamlit UI setup
st.title("Multimodal RAG Chatbot")
st.write("Ask anything about the loaded documents!")

# User input
user_query = st.text_input("Enter your question:")
if user_query:
    response = qa_chain.run(user_query)
    st.write("**Response:**", response)
