import streamlit as st
from PIL import Image
import os
from langchain.document_loaders import TextLoader, UnstructuredFileLoader
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.llms import HuggingFacePipeline
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

# Streamlit UI
st.title("Multimodal RAG with Qwen and FAISS")
st.write("Upload documents and images for multimodal retrieval and generation.")

# File uploaders
uploaded_files = st.file_uploader("Upload documents (PDF, TXT, etc.)", type=["txt", "pdf"], accept_multiple_files=True)
uploaded_images = st.file_uploader("Upload images (JPG, PNG, etc.)", type=["jpg", "png"], accept_multiple_files=True)
query = st.text_input("Enter your query:")

# Process files and images
if uploaded_files or uploaded_images:
    # Load documents
    documents = []
    for uploaded_file in uploaded_files:
        file_path = os.path.join("/tmp", uploaded_file.name)
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        if uploaded_file.name.endswith(".txt"):
            loader = TextLoader(file_path)
        else:
            loader = UnstructuredFileLoader(file_path)
        documents.extend(loader.load())

    # Load images (for now, we'll just save them; you can add image processing later)
    image_paths = []
    for uploaded_image in uploaded_images:
        image_path = os.path.join("/tmp", uploaded_image.name)
        with open(image_path, "wb") as f:
            f.write(uploaded_image.getbuffer())
        image_paths.append(image_path)

    # Initialize embeddings and vector store
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vector_store = FAISS.from_documents(documents, embeddings)

    # Initialize Qwen LLM
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen-7B-Instruct")
    model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen-7B-Instruct")
    pipe = pipeline("text-generation", model=model, tokenizer=tokenizer, max_length=512)
    llm = HuggingFacePipeline(pipeline=pipe)

    # Create RetrievalQA chain
    qa_chain = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=vector_store.as_retriever())

    # Query the system
    if query:
        response = qa_chain.run(query)
        st.write("Response:", response)