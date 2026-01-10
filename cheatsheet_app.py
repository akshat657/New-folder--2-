import streamlit as st
from typing import List,Optional
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from io import BytesIO
from reportlab.lib.pagesizes import A4
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet
import markdown2
from langchain_groq import ChatGroq
import re

load_dotenv()

LLM_MODEL = "llama-3.3-70b-versatile"
MAX_PAGES_PER_PDF = 8

SUBJECT_CATEGORIES = {
    "Mathematics": "Math (formulas, theorems, key topics)",
    "Physics": "Physics (formulas, laws, definitions)",
    "Chemistry": "Chemistry (reactions, formulas, concepts)",
    "English": "English (key ideas, definitions, analysis)",
    "History": "History (dates, events, people, significance)",
    "Biology": "Biology (terms, processes, definitions)",
    "Computer Science": "CS (algorithms, concepts, complexity analysis)",
    "Other": "General (concise summary for revision)",
}


def extract_pdf_text(pdf_files: List) -> str:
    """Extract text from uploaded PDF files"""
    output = []
    for upload in pdf_files:
        try:
            reader = PdfReader(upload)
            
            # Check page count
            if len(reader. pages) > MAX_PAGES_PER_PDF:
                st.warning(f"⚠️ {upload. name} has {len(reader.pages)} pages.  Only first {MAX_PAGES_PER_PDF} will be processed.")
            
            for page in reader.pages[: MAX_PAGES_PER_PDF]:
                text = page.extract_text()
                if text:
                    output.append(text)
        except Exception as e:
            st.error(f"Error reading {upload.name}: {str(e)}")
    
    return "\n\n".join(output)


def summarize_cheatsheet(subject: str, content: str) -> str:
    """Generate cheat sheet using AI"""
    template = f"""
You are an expert tutor creating a concise cheat sheet for {subject}. 

Subject Guidelines:  {SUBJECT_CATEGORIES.get(subject, "")}

Create a well-organized, easy-to-scan cheat sheet:  
- Use clear headings and bullet points
- Include key formulas, definitions, and concepts
- Keep it concise but comprehensive
- Format with Markdown for readability
- Aim for 1-2 pages of content

Content to summarize:
{content[: 8000]}

Generate the cheat sheet now: 
"""
    
    try:  
        model = ChatGroq(model=LLM_MODEL, temperature=0.25)
        response = model.invoke(template)
        
        # Extract clean text
        text = response.content if hasattr(response, 'content') else str(response)
        text = str(text)
        
        # Clean up formatting
        text = text.replace("\\n", "\n").replace("\\t", "    ")
        text = re.sub(r"\s*additional_kwargs=.*$", "", text, flags=re.MULTILINE)
        text = re.sub(r"\s*response_metadata=.*$", "", text, flags=re.MULTILINE)
        
        return text. strip()
        
    except Exception as e:  
        error_message = str(e)
        
        if "rate_limit" in error_message. lower() or "429" in error_message:  
            st.error("🚫 Rate limit exceeded. Please wait a moment and try again.")
        else:
            st.error(f"❌ Error generating cheat sheet: {error_message}")
        
        return ""


def generate_pdf(markdown_text: str) -> Optional[BytesIO]:
    try:
        buffer = BytesIO()
        doc = SimpleDocTemplate(buffer, pagesize=A4)
        styles = getSampleStyleSheet()
        story = []

        html = markdown2.markdown(markdown_text)

        for line in html.split("\n"):
            if line.strip():
                clean_line = re.sub("<[^<]+?>", "", line)
                if clean_line.strip():
                    story.append(Paragraph(clean_line, styles["Normal"]))
                    story.append(Spacer(1, 6))

        doc.build(story)
        buffer.seek(0)
        return buffer

    except Exception as e:
        st.error(f"Error generating PDF: {str(e)}")
        return None

def run_app():
    """Main cheatsheet application"""
    st.markdown("## 📝 Cheat Sheet Generator")
    st.markdown("*Generate concise study guides from PDFs or topics*")
    
    # Subject selection
    subject = st.selectbox(
        "📚 Select Subject:",
        list(SUBJECT_CATEGORIES.keys()),
        help="Choose the subject area for your cheat sheet"
    )
    
    # Input method tabs
    tab1, tab2 = st.tabs(["📄 From PDF", "💭 From Topic"])
    
    with tab1:
        st.markdown("### Upload Lecture PDFs")
        pdf_docs = st.file_uploader(
            f"Upload PDF files (max {MAX_PAGES_PER_PDF} pages each)",
            type="pdf",
            accept_multiple_files=True,
            key="pdf_upload"
        )
        
        if pdf_docs:
            total_pages = sum(len(PdfReader(pdf).pages) for pdf in pdf_docs)
            st.info(f"📊 Uploaded {len(pdf_docs)} file(s) with {total_pages} total pages")
    
    with tab2:
        st.markdown("### Enter Topic or Keywords")
        topic = st.text_area(
            "Describe the topic you want a cheat sheet for:",
            placeholder="e.g., Quadratic equations, derivatives, integration formulas",
            height=150,
            key="topic_input"
        )
    
    # Generate button
    st.markdown("---")
    
    if st.button("🚀 Generate Cheat Sheet", use_container_width=True, type="primary"):
        # Validate input
        has_pdf = pdf_docs is not None and len(pdf_docs) > 0
        has_topic = topic is not None and topic.strip() != ""
        
        if not has_pdf and not has_topic:
            st.error("Please upload PDFs or enter a topic.")
            return
        
        with st.spinner("🤖 Generating your cheat sheet..."):
            # Prepare content
            if has_pdf:
                text_content = extract_pdf_text(pdf_docs)
                if not text_content.strip():
                    st.error("Could not extract text from PDFs.")
                    return
                source_label = f"uploaded PDF{'s' if len(pdf_docs) > 1 else ''}"
            else:
                text_content = f"Create a comprehensive cheat sheet about:  {topic}"
                source_label = "your topic"
            
            # Generate cheat sheet
            cheatsheet = summarize_cheatsheet(subject, text_content)
            
            if not cheatsheet. strip():
                st.warning("⚠️ Could not generate cheat sheet. Please try again.")
                return
            
            # Display success
            st.success(f"✅ Cheat sheet generated from {source_label}!")
            
            # Display cheat sheet
            st.markdown("---")
            st.markdown("### 📋 Your Cheat Sheet")
            st.markdown(cheatsheet)
            
            # Download options
            st.markdown("---")
            st.markdown("### 💾 Download Options")
            
            download_col1, download_col2 = st.columns(2)
            
            with download_col1:
                # Download as markdown
                st.download_button(
                    label="📄 Download as Markdown",
                    data=cheatsheet,
                    file_name=f"cheatsheet_{subject. lower().replace(' ', '_')}.md",
                    mime="text/markdown",
                    use_container_width=True
                )
            
            with download_col2:
                # Download as PDF
                pdf_file = generate_pdf(cheatsheet)
                if pdf_file:
                    st.download_button(
                        label="📕 Download as PDF",
                        data=pdf_file,
                        file_name=f"cheatsheet_{subject.lower().replace(' ', '_')}.pdf",
                        mime="application/pdf",
                        use_container_width=True
                    )


if __name__ == "__main__":
    run_app()