import streamlit as st
import cheatsheet_app
import pdf_qa_app
import yt_summary_app

st.set_page_config(
    page_title="AI Study Assistant", 
    page_icon="🧠",
    layout="centered",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Rubik:wght@400;600&display=swap');

        html, body, [class*="css"] {
            font-family: 'Rubik', sans-serif;
        }
       
        .main {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            padding: 2rem;
        }
        
        .stApp {
            background: linear-gradient(to right, #FFDEE9, #B5FFFC);
        }
       
        .title {
            font-size: 2.8rem;
            font-weight: 700;
            color: #4b0082;
            text-align: center;
            margin-bottom: 1rem;
            text-shadow:  2px 2px 4px rgba(0,0,0,0.1);
        }
        
        .subtitle {
            font-size: 1.2rem;
            color: #444;
            text-align: center;
            margin-bottom: 2rem;
        }
    </style>
""", unsafe_allow_html=True)

# Header
st.markdown("<div class='title'>🧠 Last Minute Prep</div>", unsafe_allow_html=True)
st.markdown("<div class='subtitle'>One platform.  Three powerful AI tools for students. </div>", unsafe_allow_html=True)

# Tool Selection
st.markdown("---")
task = st.selectbox(
    "📚 Select a tool to use:",
    [
        "Cheat Sheet Generator",
        "PDF Question Answering",
        "YouTube Summarizer"
    ],
    help="Choose the AI tool that best fits your study needs"
)

# Display info about selected tool
tool_descriptions = {
    "Cheat Sheet Generator": "📝 Generate concise study guides from lecture PDFs or topics",
    "PDF Question Answering": "💬 Ask questions about your PDF documents and get instant answers",
    "YouTube Summarizer": "🎥 Get summaries of educational YouTube videos"
}

st.info(tool_descriptions[task])
st.markdown("---")

# Route to appropriate app
if task == "Cheat Sheet Generator":
    cheatsheet_app.run_app()

elif task == "PDF Question Answering":
    pdf_qa_app.run_app()

elif task == "YouTube Summarizer":  
    yt_summary_app.run_app()

# Footer
st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: gray; padding: 1rem;'>"
    "Built with ❤️ using LangChain, Groq, and Streamlit"
    "</div>",
    unsafe_allow_html=True
)