# yt_summary_app.py

import os
import re
import streamlit as st
from typing import List
from urllib.parse import urlparse, parse_qs

from dotenv import load_dotenv
from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled, NoTranscriptFound

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.prompts import PromptTemplate
from langchain.chains.summarize import load_summarize_chain
from langchain_core.documents.base import Document
from langchain_groq import ChatGroq  # ✅ Latest LangChain‑Groq integration

# — Load Groq API key from .env
load_dotenv()
if not os.getenv("GROQ_API_KEY"):
    raise ValueError("Missing GROQ_API_KEY environment variable")

map_prompt = PromptTemplate(
    input_variables=["text"],
    template=(
        "🔍 Here’s a transcript section (~500 words):\n\n"
        "{text}\n\n"
        "Summarize the key points as bullet points (≤250 words)."
    ),
)
combine_prompt = PromptTemplate(
    input_variables=["text"],
    template=(
        "📚 Now combine all these bullet lists into one coherent summary:\n\n{text}"
    ),
)

llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    temperature=0.25,
    max_retries=2  # Optional, but helpful
)

summarizer = load_summarize_chain(
    llm=llm,
    chain_type="map_reduce",
    map_prompt=map_prompt,
    combine_prompt=combine_prompt,
    verbose=False
)

def extract_video_id(url_or_id: str) -> str | None:
    s = url_or_id.strip()
    if re.fullmatch(r"[A-Za-z0-9_-]{11}", s):
        return s
    parsed = urlparse(s)
    h = (parsed.netloc or "").lower()
    if "youtu.be" in h:
        return parsed.path.lstrip("/").split("?")[0]
    if "youtube.com" in h:
        if parsed.path == "/watch":
            return parse_qs(parsed.query).get("v", [""])[0]
        if parsed.path.startswith("/live/"):
            return parsed.path.split("/live/")[-1].split("?")[0]
        m = re.search(r"/(?:embed|v)/([A-Za-z0-9_-]{11})", parsed.path)
        if m:
            return m.group(1)
    return None

def extract_transcript(url: str, languages: List[str] = ["en", "hi"]) -> str:
    vid = extract_video_id(url)
    if not vid:
        st.error("❌ Could not parse a valid YouTube video ID.")
        return ""
    try:
        fetched = YouTubeTranscriptApi().fetch(video_id=vid, languages=languages)
    except TranscriptsDisabled:
        st.error("🎙️ Captions disabled or unavailable for this video.")
        return ""
    except NoTranscriptFound:
        st.error("❓ Transcript not found for requested languages.")
        return ""
    except Exception as e:
        st.error(f"Unexpected error fetching transcript: {e}")
        return ""
    return " ".join(snippet.text.strip() for snippet in fetched)

def summarize_transcript(transcript: str) -> str:
    splitter = RecursiveCharacterTextSplitter(chunk_size=4000, chunk_overlap=400)
    docs = [Document(page_content=chunk) for chunk in splitter.split_text(transcript)]
    result = summarizer.invoke({"input_documents": docs})
    return result["output_text"].strip()

def run_app():
    st.set_page_config(page_title="YouTube → Summary (LLaMA/Groq)", layout="centered")
    st.title("YouTube Video → 📝 Smart Summary (LLaMA Cloud)")

    youtube_input = st.text_input("YouTube URL or Video ID:")
    if youtube_input and st.button("Fetch & Summarize"):
        transcript = extract_transcript(youtube_input)
        if not transcript:
            return
        with st.spinner("🔎 Summarizing with LLaMA…"):
            summary = summarize_transcript(transcript)
        if summary:
            st.markdown("## 📝 Summary")
            st.write(summary)
        else:
            st.error("⚠️ No summary generated")

if __name__ == "__main__":
    run_app()
