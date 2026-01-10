import streamlit as st
from PyPDF2 import PdfReader
from dotenv import load_dotenv

from langchain_groq import ChatGroq
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import InMemoryVectorStore

# ========= LangChain Compatibility Imports =========
try:
    from langchain.text_splitter import RecursiveCharacterTextSplitter
except ImportError:
    from langchain_text_splitters import RecursiveCharacterTextSplitter

try:
    from langchain.chains.question_answering import load_qa_chain
except ImportError:
    from langchain.chains import load_qa_chain

from langchain.prompts import PromptTemplate
# ==================================================

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
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=10000,
        chunk_overlap=1000
    )
    return splitter.split_text(text)


def get_vector_store(text_chunks):
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    return InMemoryVectorStore.from_texts(
        texts=text_chunks,
        embedding=embeddings
    )


def get_conversational_chain():
    prompt_template = """
Answer the question as detailed as possible from the provided context.
If the answer is not in the context, say:
"Answer is not present in the given PDF."

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
    if "vectorstore" not in st.session_state:
        st.warning("⚠️ Please upload and process PDFs first.")
        return

    db = st.session_state.vectorstore
    docs = db.similarity_search(user_question)
    chain = get_conversational_chain()

    response = chain(
        {"input_documents": docs, "question": user_question},
        return_only_outputs=True
    )

    st.markdown("### 💡 Answer:")
    st.write(response["output_text"])


def run_app():
    st.header("💬 Chat with Multiple PDFs using LLaMA (Groq)")

    pdf_docs = st.file_uploader(
        "📂 Upload your PDF files",
        accept_multiple_files=True,
        type=["pdf"]
    )

    if st.button("Submit & Process PDFs", use_container_width=True):
        if not pdf_docs:
            st.warning("Please upload at least one PDF.")
        else:
            with st.spinner("Processing PDFs..."):
                raw_text = get_pdf_text(pdf_docs)

                if not raw_text.strip():
                    st.error("No text extracted.")
                    return

                text_chunks = get_text_chunks(raw_text)
                st.session_state.vectorstore = get_vector_store(text_chunks)
                st.success("✅ Done! Ask questions now.")

    st.markdown("---")

    user_question = st.text_input("🗣️ Ask a question:")

    if user_question:
        user_input(user_question)
