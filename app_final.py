import streamlit as st
import cheatsheet_app
import pdf_qa_app
import yt_summary_app

st.set_page_config(page_title="AI Study Assistant", layout="centered")

st.markdown("""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Rubik:wght@400;600&display=swap');

        html, body, [class*="css"]  {
            font-family: 'Rubik', sans-serif;
            background: linear-gradient(to right, #FFDEE9, #B5FFFC);
            color: #333;
        }
       
        .title {
            font-size: 2.8rem;
            font-weight: 700;
            color: #4b0082;
            text-align: center;
            margin-bottom: 1rem;
        }
        .sub {
            font-size: 1.2rem;
            color: #444;
            text-align: center;
            margin-bottom: 2rem;
        }
        .stSelectbox label {
            font-weight: 600;
            font-size: 1.1rem;
            color: #333;
        }
        .stButton > button {
            border-radius: 10px;
            background: linear-gradient(to right, #ff6a00, #ee0979);
            color: white;
            font-weight: 600;
            padding: 0.7rem 1.5rem;
            font-size: 1rem;
            border: none;
            cursor: pointer;
        }
        .stButton > button:hover {
            background: linear-gradient(to right, #ee0979, #ff6a00);
        }
    </style>
""", unsafe_allow_html=True)

st.markdown("<div class='main-card'>", unsafe_allow_html=True)

st.markdown("<div class='title'>🧠 Last Minute Prep</div>", unsafe_allow_html=True)

st.markdown("<div class='sub'>One platform. Three powerful tools.</div>", unsafe_allow_html=True)

task = st.selectbox("Select a task to perform:", ["Cheat Sheet Generator", "PDF Question Answering", "YouTube Summarizer"])

st.markdown("---")

if task == "Cheat Sheet Generator":
    cheatsheet_app.run_app()

elif task == "PDF Question Answering":
    pdf_qa_app.run_app()

elif task == "YouTube Summarizer":
    yt_summary_app.run_app()

st.markdown("</div>", unsafe_allow_html=True)