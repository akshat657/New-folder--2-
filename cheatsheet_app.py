import streamlit as st
from typing import List
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from streamlit.runtime.uploaded_file_manager import UploadedFile
from io import BytesIO
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, ListFlowable, ListItem
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib import colors
import markdown2  # pip install markdown2
from langchain_groq import ChatGroq  # ✅ Changed from GoogleGenerativeAI
import re
load_dotenv()  # Load GROQ_API_KEY from .env

LLM_MODEL = "llama-3.3-70b-versatile"
llm = ChatGroq(model=LLM_MODEL, temperature=0.25)  # ✅ Use Groq LLaMA 3

SUBJECT_CATEGORIES = {
    "Mathematics": "Math (formulas, theorems, and important topics)",
    "Physics": "Physics (key formulas, laws, definitions)",
    "Chemistry": "Chemistry (reactions, formulas, periodic table highlights)",
    "English": "English (explanatory, key ideas, definitions)",
    "History": "History (dates, events, people)",
    "Biology": "Biology (terms, processes, definitions)",
    "Computer Science & IT": "Computer Science & IT (concise explanations of concepts, algorithms, key definitions, step-by-step logic when needed) and time complexity if present",
    "Other": "General (concise summary for revision)",
}


def extract_pdf_text(pdf_files: List[UploadedFile]) -> str:
    output = []
    for upload in pdf_files:
        reader = PdfReader(upload)
        for page in reader.pages:
            text = page.extract_text()
            if text:
                output.append(text)
    return "\n\n".join(output)

def summarize_cheatsheet(subject: str, content: str) -> str:
    template = f"""
You are an expert tutor creating a *cheat sheet* for quick revision on *{subject}*.

Use the content below to create a concise, 1-page easy-to-read cheat sheet. {SUBJECT_CATEGORIES.get(subject, "")}

- Focus on aligned bullet points with consistent indentation.
- Avoid numbering or section headings unless necessary.
- Keep the cheat sheet readable and no more than 4 sides of A4.
- Format using Markdown.

Content:

{content}

Cheat sheet (begin below):
"""
    try:
        response = llm.invoke(template)

        # Step‑1: extract the clean text
        text = getattr(response, "content", str(response))

        # Step‑2: unescape literal \n and \t if they appear
        text = text.replace("\\n", "\n").replace("\\t", "    ")

        # Step‑3: strip any trailing metadata block
        text = re.sub(r"\s*additional_kwargs=.*$", "", text)
        text = re.sub(r"\s*response_metadata=.*$", "", text)

        return text.strip()

    except Exception as e:
        error_message = str(e)

        if "rate_limit_exceeded" in error_message or "Error code: 429" in error_message:
            st.error("🚫 You've hit the daily token limit for Groq (100,000 tokens/day on free tier). Please try again later or upgrade your plan.")
        else:
            st.error(f"❌ Unexpected error: {error_message}")
        return ""
def generate_pdf(markdown_text: str) -> BytesIO:
    buffer = BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=A4)
    styles = getSampleStyleSheet()
    story = []

    # Convert markdown → HTML → reportlab Paragraph
    html = markdown2.markdown(markdown_text)

    # Break into lines and convert
    for line in html.splitlines():
        if line.strip():
            story.append(Paragraph(line, styles["Normal"]))
            story.append(Spacer(1, 6))

    doc.build(story)
    buffer.seek(0)
    return buffer
def run_app():
    st.set_page_config(page_title="Cheat Sheet Generator", layout="centered")
    st.title("🧾 Cheat Sheet Generator (PDF or Topic)")

    st.markdown("""
        <style>
            [data-testid="stMarkdownContainer"] ul {
                list-style-position: inside;
            }
        </style>
    """, unsafe_allow_html=True)

    subject = st.selectbox(
        "Subject for Cheat Sheet:", list(SUBJECT_CATEGORIES.keys()), index=0
    )
    pdf_docs = st.file_uploader(
        "Upload lecture PDF(s) — max 7 pages each",
        type="pdf",
        accept_multiple_files=True
    )
    topic = st.text_input(
        "Enter a topic or keywords for cheat sheet generation"
    )

    if st.button("Generate Cheat Sheet"):
        with st.spinner("Creating cheat sheet…"):
            text_content = ""
            source_label = ""

            if pdf_docs:
                for upload in pdf_docs:
                    reader = PdfReader(upload)
                    pages = len(reader.pages)
                    if pages > 8:
                        st.error(
                            f"Your uploaded file “{upload.name}” has {pages} pages. "
                            "Only up to 8 pages are allowed per file."
                        )
                        return
                text_content = extract_pdf_text(pdf_docs)
                if not text_content.strip():
                    st.error("Could not extract any text from uploaded PDFs.")
                    return
                source_label = "lecture PDFs"
            elif topic:
                text_content = f"Create a detailed cheat sheet on the topic: {topic}"
                source_label = "your topic input"
            else:
                st.error("Please upload PDFs or enter a topic.")
                return

            cheatsheet = summarize_cheatsheet(subject, text_content)

            if not cheatsheet.strip():
                st.warning("⚠ Cheat sheet could not be generated: try after some time")
                return

            st.success(f"✅ Cheat sheet generated using {source_label}:")
            lines = cheatsheet.splitlines()
            mid = len(lines) // 2
            part1 = "\n".join(lines[:mid])
            part2 = "\n".join(lines[mid:])

            col1, col2 = st.columns(2)
            with col1:
                st.markdown(part1)
            with col2:
                st.markdown(part2)

            st.markdown("---")
            st.markdown("✅ (You can copy this or print directly.)")

            pdf_file = generate_pdf(cheatsheet)
            st.download_button(
                label="📄 Download as PDF",
                data=pdf_file,
                file_name="cheatsheet.pdf",
                mime="application/pdf"
            )

if __name__ == "_main_":
    run_app()