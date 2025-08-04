import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_groq import ChatGroq
from langchain_community.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate

from langchain_google_genai import GoogleGenerativeAIEmbeddings

# Load API Key from .env file
from dotenv import load_dotenv
load_dotenv() 

def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            content = page.extract_text()
            if content:
                text += content
    return text

def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    return text_splitter.split_text(text)

def get_vector_store(text_chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("Faiss_index")

def get_conversational_chain():
    prompt_template = """
    Answer the question as detailed as possible from the provided context.
    If the answer is not in the context, say: "Answer is not present in the given PDF."

    Context:
    {context}

    Question:
    {question}

    Answer:
    """
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    model = ChatGroq(model="llama-3.3-70b-versatile", temperature=0.3)
    chain = load_qa_chain(llm=model, chain_type="stuff", prompt=prompt)
    return chain

def user_input(user_question):
    try:
        embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        db = FAISS.load_local("Faiss_index", embeddings, allow_dangerous_deserialization=True)
        docs = db.similarity_search(user_question)
        chain = get_conversational_chain()
        response = chain({"input_documents": docs, "question": user_question}, return_only_outputs=True)
        st.write("Reply:", response["output_text"])
    except Exception as e:
        st.error(f"Error: {e}")

def run_app():
    st.set_page_config(page_title="Chat With Multiple PDFs", layout="centered")
    st.header("Chat with Multiple PDFs using LLaMA (Groq)")

    # — Upload UI on the main page, not in sidebar:
    pdf_docs = st.file_uploader(
        "📂 Upload your PDF files (you can upload multiple)",
        accept_multiple_files=True,
        type=["pdf"]
    )
    if st.button("Submit & Process PDFs"):
        if not pdf_docs:
            st.warning("Please upload at least one PDF before proceeding.")
        else:
            with st.spinner("Processing PDFs..."):
                raw_text   = get_pdf_text(pdf_docs)
                text_chunks = get_text_chunks(raw_text)
                get_vector_store(text_chunks)
                st.success("Done! You can now ask questions.")

    st.markdown("---")  # separator before chat input

    user_question = st.text_input("🗣️ Ask a question based on the uploaded PDF content:")
    if user_question:
        user_input(user_question)

if __name__ == "__main__":
    run_app()
