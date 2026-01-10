import streamlit as st
from typing import List, Optional
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from io import BytesIO
from reportlab.lib.pagesizes import A4
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet
import markdown2
import re

from langchain_groq import ChatGroq

# ========= LangChain Compatibility =========
try:
    from langchain.text_splitter import RecursiveCharacterTextSplitter
except:
    from langchain_text_splitters import RecursiveCharacterTextSplitter

try:
    from langchain.chains.summarize import load_summarize_chain
except:
    from langchain.chains import load_summarize_chain
# ==========================================

load_dotenv()

LLM_MODEL = "llama-3.3-70b-versatile"
MAX_PAGES_PER_PDF = 8

SUBJECT_CATEGORIES = {
    "Mathematics": "Math formulas & theorems",
    "Physics": "Physics laws & numericals",
    "Chemistry": "Reactions & concepts",
    "English": "Key ideas & analysis",
    "History": "Dates & events",
    "Biology": "Processes & terms",
    "Computer Science": "Algorithms & complexity",
    "Other": "General revision",
}


def extract_pdf_text(pdf_files: List) -> str:
    output = []
    for upload in pdf_files:
        reader = PdfReader(upload)
        for page in reader.pages[:MAX_PAGES_PER_PDF]:
            text = page.extract_text()
            if text:
                output.append(text)
    return "\n\n".join(output)


def summarize_cheatsheet(subject: str, content: str) -> str:
    prompt = f"""
Create a concise cheat sheet for {subject}.
Use headings, bullet points and formulas.
Content:
{content[:8000]}
"""

    model = ChatGroq(model=LLM_MODEL, temperature=0.25)
    response = model.invoke(prompt)
    return str(response.content).strip()


def generate_pdf(markdown_text: str) -> Optional[BytesIO]:
    buffer = BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=A4)
    styles = getSampleStyleSheet()
    story = []

    html = markdown2.markdown(markdown_text)
    for line in html.split("\n"):
        clean = re.sub("<[^<]+?>", "", line)
        if clean.strip():
            story.append(Paragraph(clean, styles["Normal"]))
            story.append(Spacer(1, 6))

    doc.build(story)
    buffer.seek(0)
    return buffer


def run_app():
    st.markdown("## 📝 Cheat Sheet Generator")

    subject = st.selectbox("📚 Subject", SUBJECT_CATEGORIES.keys())

    tab1, tab2 = st.tabs(["📄 From PDF", "💭 From Topic"])

    with tab1:
        pdf_docs = st.file_uploader(
            "Upload PDFs",
            type="pdf",
            accept_multiple_files=True
        )

    with tab2:
        topic = st.text_area("Enter topic")

    if st.button("🚀 Generate Cheat Sheet", use_container_width=True):
        if pdf_docs:
            content = extract_pdf_text(pdf_docs)
        else:
            content = f"Create cheat sheet on {topic}"

        cheatsheet = summarize_cheatsheet(subject, content)

        st.markdown("### 📋 Cheat Sheet")
        st.markdown(cheatsheet)

        st.download_button(
            "📄 Download Markdown",
            cheatsheet,
            file_name="cheatsheet.md"
        )

        pdf_file = generate_pdf(cheatsheet)
        if pdf_file:
            st.download_button(
                "📕 Download PDF",
                pdf_file,
                file_name="cheatsheet.pdf"
            )


if __name__ == "__main__":
    run_app()
