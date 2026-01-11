import streamlit as st
from typing import List, Optional, Dict
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from io import BytesIO
from reportlab.lib.pagesizes import A4
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet
import markdown2
import re

from langchain_groq import ChatGroq
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

# ========= LangChain Compatibility =========
try:
    from langchain.text_splitter import RecursiveCharacterTextSplitter
except:   
    from langchain_text_splitters import RecursiveCharacterTextSplitter

try:  
    from langchain.chains. summarize import load_summarize_chain
except:  
    from langchain. chains import load_summarize_chain

try:  
    from langchain.chains.question_answering import load_qa_chain
except:  
    from langchain.chains import load_qa_chain

try:  
    from langchain.prompts import PromptTemplate
except: 
    from langchain.prompts. prompt import PromptTemplate
# ==========================================

load_dotenv()

LLM_MODEL = "llama-3.3-70b-versatile"
MAX_PAGES_PER_PDF = 8

SUBJECT_CATEGORIES = {
    "Mathematics":   "Math formulas & theorems",
    "Physics":  "Physics laws & numericals",
    "Chemistry": "Reactions & concepts",
    "English": "Key ideas & analysis",
    "History": "Dates & events",
    "Biology": "Processes & terms",
    "Computer Science": "Algorithms & complexity",
    "Other": "General revision",
}


def extract_pdf_text(pdf_files:  List) -> str:
    output = []
    for upload in pdf_files:  
        reader = PdfReader(upload)
        for page in reader.pages[: MAX_PAGES_PER_PDF]:
            text = page.extract_text()
            if text:  
                output.append(text)
    return "\n\n".join(output)


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
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("Faiss_index")


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
        model=LLM_MODEL,
        temperature=0.3
    )

    chain = load_qa_chain(
        llm=model,
        chain_type="stuff",
        prompt=prompt
    )
    return chain


def summarize_cheatsheet(subject: str, content: str, subtopics:   str, num_pages: int) -> str:
    prompt = f"""
Create a {num_pages}-page cheat sheet for {subject}. 
Focus on these subtopics:  {subtopics}
Use headings, bullet points, formulas, and key concepts.   
Make it concise but comprehensive for exam prep.  

Content:
{content[: 8000]}
"""

    model = ChatGroq(model=LLM_MODEL, temperature=0.25)
    response = model.invoke(prompt)
    return str(response.content).strip()


def generate_quiz(subject: str, content: str, num_questions: int) -> str:
    prompt = f"""
Generate {num_questions} multiple-choice questions for {subject} based on this content.   
Format each question EXACTLY as shown:  

Q1. [Question text here]
A) [Option A]
B) [Option B]
C) [Option C]
D) [Option D]
Answer: A

Q2. [Question text here]
A) [Option A]
B) [Option B]
C) [Option C]
D) [Option D]
Answer: B

Make sure each question follows this exact format with "Answer: " followed by the letter only.  

Content:
{content[: 8000]}
"""

    model = ChatGroq(model=LLM_MODEL, temperature=0.3)
    response = model.invoke(prompt)
    return str(response.content).strip()


def parse_quiz(quiz_text: str) -> List[Dict]:
    """Parse quiz text into structured format"""
    questions = []
    current_question = {}
    
    lines = quiz_text.strip().split('\n')
    
    for line in lines:
        line = line.strip()
        if not line:
            continue
            
        # Match question number
        if re.match(r'^Q\d+\.', line):
            if current_question:
                questions. append(current_question)
            current_question = {
                'question': re.sub(r'^Q\d+\.\s*', '', line),
                'options': {},
                'answer': ''
            }
        # Match options
        elif re.match(r'^[A-D]\)', line):
            option_letter = line[0]
            option_text = line[3:].strip()
            if current_question:
                current_question['options'][option_letter] = option_text
        # Match answer
        elif line.startswith('Answer:') or line.startswith('**Answer:'):
            answer_text = re.sub(r'\*\*Answer:\s*|\*\*|Answer:\s*', '', line).strip()
            # Extract just the letter
            answer_match = re.search(r'[A-D]', answer_text)
            if answer_match and current_question:
                current_question['answer'] = answer_match.group(0)
    
    # Add last question
    if current_question and current_question. get('question'):
        questions.append(current_question)
    
    return questions


def generate_mnemonics(subject: str, content: str) -> str:
    prompt = f"""
Create memory tricks, mnemonics, or short stories to help remember key concepts in {subject}. 
Make them fun, relatable, and easy to recall during exams.  

Content:
{content[:6000]}
"""

    model = ChatGroq(model=LLM_MODEL, temperature=0.5)
    response = model.invoke(prompt)
    return str(response.content).strip()


def generate_important_questions(subject:   str, content: str, num_questions: int) -> str:
    prompt = f"""
List {num_questions} potential exam questions for {subject} based on this content.   
Include short answer, long answer, and application-based questions.  

Content:
{content[:8000]}
"""

    model = ChatGroq(model=LLM_MODEL, temperature=0.3)
    response = model. invoke(prompt)
    return str(response.content).strip()


def user_input(user_question):
    try:
        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )

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


def generate_pdf(markdown_text: str) -> Optional[BytesIO]:
    buffer = BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=A4)
    styles = getSampleStyleSheet()
    story = []

    html = markdown2.markdown(markdown_text)
    for line in html.split("\n"):
        clean = re.sub("<[^<]+?>", "", line)
        if clean. strip():
            story.append(Paragraph(clean, styles["Normal"]))
            story.append(Spacer(1, 6))

    doc.build(story)
    buffer.seek(0)
    return buffer


def run_app():
    st.markdown("## 📝 Study Assistant - Cheat Sheets & PDF Q&A")

    # Initialize session state
    if 'selected_action' not in st.session_state:
        st.session_state.selected_action = None
    if 'quiz_mode' not in st.session_state:
        st.session_state.quiz_mode = None  # 'interactive' or 'download'
    if 'quiz_data' not in st.session_state:
        st.session_state.quiz_data = None
    if 'user_answers' not in st.session_state:
        st.session_state.user_answers = {}
    if 'quiz_submitted' not in st.session_state:
        st.session_state.quiz_submitted = False

    # Input method selection
    input_mode = st.radio("Choose input method:", ["📄 Upload PDF", "💭 Enter Topic"], horizontal=True)

    pdf_docs = None
    topic = None
    subject = None

    if input_mode == "📄 Upload PDF":   
        pdf_docs = st.file_uploader(
            "Upload PDFs",
            type="pdf",
            accept_multiple_files=True
        )
        subject = "General Study Material"
    else:  
        topic = st.text_area("Enter your topic", placeholder="e.g., Photosynthesis, Newton's Laws, etc.")
        subject = st.selectbox("📚 Select Subject", SUBJECT_CATEGORIES.keys())

    # Common options
    st.markdown("### ⚙️ Options")
    subtopics = st.text_input("Specific subtopics (optional)", placeholder="e.g., Definitions, Formulas, Examples")

    # Action buttons
    st.markdown("### 🚀 What do you want to do?")
    
    # Create buttons based on whether PDF is uploaded
    if input_mode == "📄 Upload PDF" and pdf_docs: 
        col1, col2, col3, col4, col5 = st.columns(5)
        
        with col1:
            if st.button("📋 Cheat Sheet", use_container_width=True):
                st.session_state. selected_action = "cheatsheet"
                st.session_state.quiz_mode = None
        with col2:
            if st. button("🎯 Quiz", use_container_width=True):
                st. session_state.selected_action = "quiz"
                st.session_state.quiz_mode = None
        with col3:
            if st.button("🧠 Mnemonics", use_container_width=True):
                st.session_state.selected_action = "mnemonics"
                st.session_state.quiz_mode = None
        with col4:
            if st.button("❓ Important Qs", use_container_width=True):
                st.session_state.selected_action = "questions"
                st.session_state.quiz_mode = None
        with col5:
            if st.button("💬 Ask PDF", use_container_width=True):
                st.session_state.selected_action = "pdf_qa"
                st.session_state.quiz_mode = None
    else:
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            if st.button("📋 Cheat Sheet", use_container_width=True):
                st. session_state.selected_action = "cheatsheet"
                st.session_state.quiz_mode = None
        with col2:
            if st.button("🎯 Quiz", use_container_width=True):
                st.session_state.selected_action = "quiz"
                st. session_state.quiz_mode = None
        with col3:
            if st.button("🧠 Mnemonics", use_container_width=True):
                st.session_state.selected_action = "mnemonics"
                st.session_state. quiz_mode = None
        with col4:
            if st. button("❓ Important Qs", use_container_width=True):
                st.session_state.selected_action = "questions"
                st.session_state.quiz_mode = None

    # Validate inputs
    has_valid_input = False
    content = ""
    
    if input_mode == "📄 Upload PDF" and pdf_docs:  
        has_valid_input = True
        content = extract_pdf_text(pdf_docs)
        if not subject:
            subject = "General Study Material"
    elif input_mode == "💭 Enter Topic" and topic:
        has_valid_input = True
        content = f"Create materials on:   {topic}"
        if not subject:  
            subject = "Other"

    # Show specific options based on selected action
    if st.session_state.selected_action and has_valid_input:
        st.markdown("---")
        
        if st.session_state. selected_action == "cheatsheet":  
            st.markdown("### 📋 Cheat Sheet Options")
            num_pages = st.slider("Number of pages:", 1, 5, 2, key="cheat_pages")
            
            if st.button("✨ Generate Cheat Sheet", use_container_width=True, type="primary"):
                with st.spinner("Creating your cheat sheet..."):
                    subtopics_text = subtopics if subtopics else "all major concepts"
                    result = summarize_cheatsheet(subject, content, subtopics_text, num_pages)
                    
                    st.markdown("### 📋 Your Cheat Sheet")
                    st.markdown(result)
                    
                    col1, col2 = st. columns(2)
                    with col1:
                        st. download_button("📄 Download Markdown", result, file_name="cheatsheet.md")
                    with col2:
                        pdf_file = generate_pdf(result)
                        if pdf_file:
                            st.download_button("📕 Download PDF", pdf_file, file_name="cheatsheet. pdf")
        
        elif st.session_state.selected_action == "quiz":
            st. markdown("### 🎯 Quiz Options")
            num_questions = st.slider("Number of questions:", 5, 20, 10, key="quiz_qs")
            
            # Quiz mode selection
            if st.session_state.quiz_mode is None:
                st.info("Choose how you want to use the quiz:")
                col1, col2 = st. columns(2)
                
                with col1:
                    if st.button("🎮 Take Interactive Test", use_container_width=True, type="primary"):
                        st.session_state.quiz_mode = "interactive"
                        st.rerun()
                
                with col2:
                    if st. button("📄 Download Quiz", use_container_width=True):
                        st.session_state.quiz_mode = "download"
                        st.rerun()
            
            # Interactive quiz mode
            elif st.session_state.quiz_mode == "interactive":  
                if st.session_state.quiz_data is None:
                    with st.spinner("Creating your quiz..."):
                        quiz_text = generate_quiz(subject, content, num_questions)
                        parsed_quiz = parse_quiz(quiz_text)
                        
                        if not parsed_quiz:
                            st.error("❌ Failed to parse quiz. Please try again.")
                            if st.button("🔄 Retry"):
                                st.session_state.quiz_mode = None
                                st.rerun()
                        else:
                            st. session_state.quiz_data = parsed_quiz
                            st. rerun()
                
                if st.session_state.quiz_data:
                    st.markdown("### 🎮 Interactive Quiz")
                    st.info("Select your answers and click Submit to see your score!")
                    
                    # Display all questions
                    for idx, q_data in enumerate(st.session_state.quiz_data):
                        st.markdown(f"#### Question {idx + 1}")
                        st.write(q_data['question'])
                        
                        # Create radio buttons for options
                        options_list = [f"{k}) {v}" for k, v in q_data['options'].items()]
                        
                        if not st.session_state.quiz_submitted:
                            selected = st.radio(
                                f"Select your answer:",
                                options_list,
                                key=f"q_{idx}",
                                index=None
                            )
                            if selected:
                                st.session_state.user_answers[idx] = selected[0]  # Store the letter (A, B, C, or D)
                        else:
                            # Show results
                            user_ans = st.session_state.user_answers. get(idx, "")
                            correct_ans = q_data['answer']
                            
                            for opt in options_list:
                                opt_letter = opt[0]
                                if opt_letter == correct_ans: 
                                    st.success(f"✅ {opt} (Correct Answer)")
                                elif opt_letter == user_ans and user_ans != correct_ans:  
                                    st.error(f"❌ {opt} (Your Answer)")
                                else:  
                                    st.write(opt)
                        
                        st.markdown("---")
                    
                    # Submit or restart
                    if not st.session_state.quiz_submitted:
                        if st. button("📊 Submit Quiz", use_container_width=True, type="primary"):
                            st.session_state.quiz_submitted = True
                            st.rerun()
                    else: 
                        # Calculate score
                        correct_count = 0
                        total = len(st.session_state. quiz_data)
                        
                        for idx, q_data in enumerate(st.session_state.quiz_data):
                            if st.session_state.user_answers.get(idx) == q_data['answer']: 
                                correct_count += 1
                        
                        score_percentage = (correct_count / total) * 100
                        
                        # Display score with emoji
                        st.markdown("### 🎯 Your Results")
                        
                        if score_percentage >= 80:
                            st.success(f"🎉 Excellent! You scored {correct_count}/{total} ({score_percentage:.1f}%)")
                        elif score_percentage >= 60:
                            st.info(f"👍 Good job! You scored {correct_count}/{total} ({score_percentage:.1f}%)")
                        else:
                            st.warning(f"📚 Keep practicing! You scored {correct_count}/{total} ({score_percentage:.1f}%)")
                        
                        # Restart button
                        if st.button("🔄 Take Another Quiz", use_container_width=True):
                            st.session_state.quiz_data = None
                            st.session_state.user_answers = {}
                            st.session_state.quiz_submitted = False
                            st.session_state.quiz_mode = None
                            st.rerun()
            
            # Download mode
            elif st.session_state.quiz_mode == "download":  
                with st.spinner("Creating quiz... "):
                    result = generate_quiz(subject, content, num_questions)
                    
                    st.markdown("### 🎯 Your Quiz")
                    st.markdown(result)
                    
                    st.download_button("📄 Download Quiz", result, file_name="quiz.md")
                    
                    if st.button("🔄 Back to Options"):
                        st.session_state.quiz_mode = None
                        st.rerun()
        
        elif st.session_state.selected_action == "mnemonics":
            st.markdown("### 🧠 Memory Tricks & Mnemonics")
            st.info("Creating fun memory aids to help you remember key concepts!")
            
            if st.button("✨ Generate Mnemonics", use_container_width=True, type="primary"):
                with st.spinner("Creating memory tricks..."):
                    result = generate_mnemonics(subject, content)
                    
                    st.markdown("### 🧠 Your Memory Tricks")
                    st. markdown(result)
                    
                    st.download_button("📄 Download Mnemonics", result, file_name="mnemonics. md")
        
        elif st.session_state.selected_action == "questions":  
            st.markdown("### ❓ Important Exam Questions")
            num_questions = st.slider("Number of questions:", 5, 20, 10, key="imp_qs")
            
            if st.button("✨ Generate Questions", use_container_width=True, type="primary"):
                with st.spinner("Generating important questions..."):
                    result = generate_important_questions(subject, content, num_questions)
                    
                    st.markdown("### ❓ Potential Exam Questions")
                    st.markdown(result)
                    
                    st.download_button("📄 Download Questions", result, file_name="important_questions.md")
        
        elif st.session_state.selected_action == "pdf_qa":  
            st.markdown("### 💬 Ask Your PDF")
            
            # Process PDF for Q&A if not already done
            if 'pdf_processed' not in st.session_state:
                with st.spinner("Processing PDFs for Q&A..."):
                    text_chunks = get_text_chunks(content)
                    get_vector_store(text_chunks)
                    st.session_state['pdf_processed'] = True
                    st.session_state['pdf_text'] = content
                    st.success("✅ PDF processed!   Ask your questions below.")
            
            user_question = st.text_input("🗣️ Ask a question about your PDF:")
            
            if user_question:  
                user_input(user_question)
    
    elif st.session_state.selected_action and not has_valid_input:  
        st.warning("⚠️ Please upload a PDF or enter a topic first!")


if __name__ == "__main__":  
    run_app()