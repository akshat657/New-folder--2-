import streamlit as st
from PyPDF2 import PdfReader
import streamlit as st
from PyPDF2 import PdfReader
from langchain. text_splitter import RecursiveCharacterTextSplitter
import os

from langchain_groq import ChatGroq
from langchain_community. embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain. chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
load_dotenv()


def get_pdf_text(pdf_docs):
    """Extract text from uploaded PDF files"""
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader. pages:
            content = page. extract_text()
            if content: 
                text += content
    return text


def get_text_chunks(text):
    """Split text into chunks"""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=10000,
        chunk_overlap=1000
    )
    return text_splitter.split_text(text)


def get_vector_store(text_chunks):
    """Create and save FAISS vector store with HuggingFace embeddings"""
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vector_store = FAISS. from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("Faiss_index")


def get_conversational_chain():
    """Create Q&A chain"""
    prompt_template = """
Answer the question as detailed as possible from the provided context.  
If the answer is not in the context, say:  "Answer is not present in the given PDF."

Context:  
{context}

Question: 
{question}

Answer:
"""
    
    prompt = PromptTemplate(
        template=prompt_template,
        input_variables=["context", "question"]
    )
    
    model = ChatGroq(
        model="llama-3.3-70b-versatile",
        temperature=0.3
    )
    
    chain = load_qa_chain(
        llm=model,
        chain_type="stuff",
        prompt=prompt
    )
    
    return chain


def user_input(user_question):
    """Process user question"""
    try:
        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        
        db = FAISS.load_local(
            "Faiss_index",
            embeddings,
            allow_dangerous_deserialization=True
        )
        
        docs = db.similarity_search(user_question)
        chain = get_conversational_chain()
        
        response = chain(
            {"input_documents": docs, "question": user_question},
            return_only_outputs=True
        )
        
        st.markdown("### 💡 Answer:")
        st.write(response["output_text"])
        
    except FileNotFoundError: 
        st. warning("⚠️ Please upload and process PDFs first.")
    except Exception as e:
        st.error(f"❌ Error: {str(e)}")


def run_app():
    """Main PDF Q&A application"""
    st.set_page_config(
        page_title="Chat With Multiple PDFs",
        page_icon="💬",
        layout="centered"
    )
    
    st.header("💬 Chat with Multiple PDFs using LLaMA (Groq)")
    
    # Upload section
    pdf_docs = st.file_uploader(
        "📂 Upload your PDF files (you can upload multiple)",
        accept_multiple_files=True,
        type=["pdf"]
    )
    
    if st.button("Submit & Process PDFs", use_container_width=True):
        if not pdf_docs:
            st.warning("Please upload at least one PDF before proceeding.")
        else:
            with st.spinner("Processing PDFs... "):
                # Extract text
                raw_text = get_pdf_text(pdf_docs)
                
                if not raw_text. strip():
                    st.error("No text could be extracted from PDFs.")
                    return
                
                # Create chunks
                text_chunks = get_text_chunks(raw_text)
                st.info(f"Created {len(text_chunks)} text chunks")
                
                # Create vector store
                get_vector_store(text_chunks)
                
                st.success("✅ Done! You can now ask questions.")
    
    st.markdown("---")
    
    # Question input
    user_question = st. text_input(
        "🗣️ Ask a question based on the uploaded PDF content:",
        key="question_input"
    )
    
    if user_question:
        user_input(user_question)


if __name__ == "__main__":
    run_app()