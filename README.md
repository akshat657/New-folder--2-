# 🧠 AI Study Assistant

A minimal yet powerful multi-tool Streamlit app designed to **supercharge your last-minute exam preparation**. Whether you're cramming the night before or reviewing key concepts quickly — this assistant has your back.

---

## 🎯 Ideal For

> ⚡ **Students needing fast revision**  
> 📌 **Last-minute concept brushing**  
> ✍️ **Quick notes & summaries**  
> 🧠 **Self-questioning from long PDFs**

---

## 🚀 Features

- 📚 **Cheat Sheet Generator**: Upload text, and get a compact, clear summary.
- 📄 **PDF Question Answering**: Ask questions directly from your PDF documents.
- 📹 **YouTube Summarizer**: Paste a video link and get clean, readable summaries.

---

## 🖼️ Interface Preview

![screenshot](preview.png) <!-- Replace this with your actual screenshot file or URL -->
<img width="1154" height="841" alt="Screenshot 2025-08-04 053413" src="https://github.com/user-attachments/assets/3b7fc038-eed5-4f5c-9dfe-384c4f5ec7d2" />

---

## 🎨 Tech Stack

- **Frontend**: [Streamlit](https://streamlit.io/)
- **LLMs**: Google Gemini (via LangChain)
- **Embeddings & Vector Search**: FAISS + GoogleGenerativeAIEmbeddings
- **Other Libraries**: PyMuPDF, PyPDF2, LangChain, `dotenv`, etc.

---

## 📦 Project Structure

```

├── main\_app.py              # Main multipage controller
├── cheatsheet\_app.py        # Cheat Sheet Generator module
├── pdf\_qa\_app.py            # PDF QA using LangChain + Gemini
├── yt\_summary\_app.py        # YouTube transcript summarizer
├── requirements.txt         # Python dependencies
├── .streamlit/
│   └── config.toml          # Streamlit UI configuration
└── README.md                # Project overview (this file)

````

---

## 🔧 Setup Instructions

### 1️⃣ Clone the repository

```bash
git clone https://github.com/your-username/ai-study-assistant.git
cd ai-study-assistant
````

### 2️⃣ Install dependencies

```bash
pip install -r requirements.txt
```

### 3️⃣ Add your Gemini API key in `.env` file

```env
GOOGLE_API_KEY=your_gemini_api_key
```

### 4️⃣ Run the app

```bash
streamlit run main_app.py
```

---

## 🌍 Live Demo

[![Open in Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://your-streamlit-link.com)

---

## ✨ Purpose

**Why this app?**
Because exams are scary and time is short.
This app brings **three AI tools in one place** to help students **prepare smarter, not harder — especially when the clock’s ticking**.

---

## 🤝 Contributing

Pull requests are welcome. For major changes, please open an issue first.

---

## 📄 License

This project is licensed under the **MIT License**.

---

## 👨‍💻 Author

**Akshat Khandelwal**
📧 Email: [akshatkhandelwal004@gmail.com](mailto:akshatkhandelwal004@gmail.com)
🐙 GitHub: [@akshat657](https://github.com/akshat657)

```

---

Let me know if you'd like:
- A cool **ASCII banner** at the top?
- GitHub **badges** (stars, forks, license)?
- Instructions for deploying to **Streamlit Cloud**?

I can add those too.
```