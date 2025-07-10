<img width="894" height="702" alt="AI" src="https://github.com/user-attachments/assets/f4219fe7-a1a5-4d45-820f-992842f3d9c3" />
<img width="1916" height="1080" alt="page0 fronend" src="https://github.com/user-attachments/assets/3460988c-52a6-4498-bed2-6f4763ff991f" />

https://github.com/user-attachments/assets/eca741b3-291c-4825-888c-fcc2d926e3fb

## 🧠 InsightBot.AI: Company Knowledge Assistant 🤖  
🌐 Live App 🔗 **[Launch InsightBot.AI](https://insightbot-ai-company-knowledge-assistant.streamlit.app/)**

<p align="center">
  <b>Your AI-Powered Document Chat Assistant for Smarter Company Knowledge Access</b><br>
  Upload 📄 | Ask 💬 | Cite 🔍 | Summarize ✨ | Translate 🌍 | All in One Streamlit App 🚀
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Streamlit-UI-red?style=for-the-badge&logo=streamlit" />
  <img src="https://img.shields.io/badge/AI%20Powered-Gemini%20%7C%20Groq-blueviolet?style=for-the-badge" />
  <img src="https://img.shields.io/badge/Multilingual-Translate%20Enabled-orange?style=for-the-badge" />
  <img src="https://img.shields.io/badge/ChromaDB-DuckDB%20Backend-green?style=for-the-badge" />
  <img src="https://img.shields.io/badge/Knowledge%20Bot-LangChain%20%7C%20LangGraph-yellow?style=for-the-badge" />
</p>

<p align="center">
  <img src="https://media.giphy.com/media/v1.Y2lkPTc5MGI3NjExY2l2bzdjeWpoZ25qYm13aGhxZ3h3ZXVvcHI2aHYxdTdtcHh0dTlhbiZlcD12MV9naWZzX3NlYXJjaCZjdD1n/7SF5scGB2AFrgsXP63/giphy.gif" width="70%" />
</p>



# 🧠 InsightBot.AI: Company Knowledge Assistant 🤖

🔗 **Live App**: [https://insightbot-ai-company-knowledge-assistant.streamlit.app/](https://insightbot-ai-company-knowledge-assistant.streamlit.app/)

InsightBot.AI is a powerful AI-powered knowledge assistant tailored for companies and teams. It allows users to **upload internal documents**, builds a **searchable knowledge base**, and enables smart interaction through **natural language queries** in multiple languages — all through a beautiful, modern Streamlit UI
InsightBot.AI: Company Knowledge Assistant is a modern, interactive Streamlit web application designed to serve as a custom company knowledge base chatbot. It enables users to upload multiple document types (PDF, DOCX, TXT), builds a searchable knowledge base using advanced AI (LangChain, Gemini, Groq, LangGraph), and provides an engaging chat interface for Q&A with context-aware answers and source citations. The app is visually vibrant, supports multilingual interaction, and is robustly engineered for both local and cloud deployment..

---

## 🧭 Table of Contents

- [🔍 Project Summary](#-project-summary)
- [✨ Key Features](#-key-features)
- [💡 Use Cases](#-use-cases)
- [🛠️ Tech Stack](#-tech-stack)
- [🚀 Setup Instructions](#-setup-instructions)
- [🌐 Deployment](#-deployment)
- [🧩 Architecture Overview](#-architecture-overview)
- [📂 Folder Structure](#-folder-structure)
- [📸 Screenshots (optional)](#-screenshots)
- [📜 License](#-license)
- [🤝 Acknowledgements](#-acknowledgements)

---

## 🔍 Project Summary

🧠 **InsightBot.AI** is designed to act as a custom chatbot interface that gives **contextual answers** from uploaded documents (PDF, DOCX, TXT) and provides source citations. It is built on top of modern LLM and AI frameworks like:

- **LangChain** for orchestrating prompt flows
- **ChromaDB with DuckDB** backend for vector storage
- **Groq API & Gemini (via LangChain)** for ultra-fast, intelligent responses
- **Google Translate** for real-time multilingual conversation support

All this is wrapped in a stunningly styled and fully interactive **Streamlit UI** optimized for both local and cloud deployment.

---

## ✨ Key Features

### 📁 1. **Document Upload & Management**
- Upload and manage multiple document formats:
  - PDF
  - DOCX
  - TXT
- Uses:
  - `PyPDF2` as primary parser
  - `Tika` as backup extractor
  - `docx2txt` for `.docx` support
- Supports:
  - Live preview in sidebar
  - File filtering by name/type/keyword
  - Document search & summary

---

### 🧠 2. **Knowledge Base Construction**
- Vector store built using **LangChain + ChromaDB**.
- Uses **HuggingFaceEmbeddings** (runs on CPU for universal compatibility).
- **Persistent storage**: Vectors saved in `chroma_db/` directory.
- Uses **DuckDB** instead of SQLite for full cloud support (SQLite on Streamlit Cloud is outdated).
- Automatically updates with new documents.

---

### 🤖 3. **AI-Powered Chatbot**
- Chat interface to ask document-based questions.
- Uses:
  - **Gemini LLM** via `langchain-google-genai`
  - **Groq API** for blazing fast inference
- Intelligent, context-aware answers with:
  - Source document citations
  - Expandable full source views
- Maintains complete chat history with export options:
  - Markdown
  - Plain text

---

### 📑 4. **Document Summarization & Smart Navigation**
- Uses **Gemini** to auto-summarize each uploaded document.
- Extracts important **sections or topics**.
- Sidebar features “Jump to Section” buttons for easy navigation.

---

### 💬 5. **Smart Suggestions & AI Shortcuts**
- After each answer, suggests:
  - Follow-up questions
  - Related topics
  - Suggested actions (pin, edit, preview, ask all)
- Users can:
  - Pin suggestions
  - Copy or preview suggestions
  - Track suggestion history in cards

---

### 🌐 6. **Multilingual Chat Support**
- Choose from multiple languages:
  - 🇬🇧 English
  - 🇮🇳 Hindi
  - 🇫🇷 French
  - 🇩🇪 German
  - 🇪🇸 Spanish
  - 🇨🇳 Chinese
  - 🇸🇦 Arabic
- Uses `googletrans` (Google Translate API) to:
  - Translate user input
  - Translate AI responses back to user’s language

---

### ⭐ 7. **User Feedback System**
- Rate AI answers with 👍 / 👎
- Add custom feedback comments
- Edit, delete, or regenerate messages
- Star/favorite key conversations
- Auto-detects and extracts topics from each chat

---

### 🎨 8. **Beautiful UI/UX with Animations**
- Highly customized Streamlit layout with:
  - 🎨 Colorful headers and dynamic title
  - 🧑‍💻 Developer profile + logo in sidebar
  - 💬 Chat bubbles with animated send button
  - 🌈 Smooth sidebar transitions
- Modern card layout for history, suggestions, and document previews

---

### 🧯 9. **Robust Error Handling**
- Handles:
  - Missing dependencies (e.g. `googletrans`, `langchain-community`)
  - File parsing issues
  - Model availability problems
- Includes clear error messages and fallback logic

---

### ☁️ 10. **Fully Cloud Compatible**
- Streamlit Cloud-compatible
- Uses `DuckDB` backend (instead of SQLite) for ChromaDB
- `requirements.txt` pre-configured with:
  - langchain
  - langchain-community
  - duckdb
  - groq
  - googletrans
  - and more!

---

## 💡 Use Cases

- 🏢 Internal Knowledge Base for Companies
- 📄 Legal Document Summarization & Q&A
- 📚 Research Assistant for Students
- 🧾 HR or Policy Document Chatbot
- 💬 Multi-language Support Chatbot for Teams

---

## 🛠️ Tech Stack

| Layer               | Tools Used                                                |
|--------------------|-----------------------------------------------------------|
| 🧱 Backend          | Python, LangChain, DuckDB, ChromaDB, Gemini, Groq         |
| 📦 File Parsing     | PyPDF2, Tika, docx2txt                                    |
| 🧠 LLMs             | Gemini Pro, Groq, HuggingFace                             |
| 🗂️ Vector DB        | Chroma with DuckDB (not SQLite)                           |
| 🌐 UI Framework     | Streamlit + Custom CSS                                    |
| 🌍 Translation      | googletrans==4.0.0-rc1                                     |
| ☁️ Deployment       | Streamlit Cloud                                            |

---

## 🚀 Setup Instructions (Local)

1. **Clone the Repo**
```bash
git clone https://github.com/abhishekkumar62000/InsightBot.AI--Company-Knowledge-Assistant.git
cd insightbot-ai
````

2. **Install Dependencies**

```bash
pip install -r requirements.txt
```

3. **Run App**

```bash
streamlit run App.py
```

---

## 🌐 Deployment Instructions (Streamlit Cloud)

> ✅ This app is 100% ready for deployment on Streamlit Cloud!

1. Push your code to a public GitHub repository.
2. Go to [streamlit.io/cloud](https://streamlit.io/cloud) and click “New App”.
3. Connect your GitHub repo.
4. Ensure `requirements.txt` is complete (use provided version).
5. Click “Deploy” and you’re live!

**Pro Tip**: Use `duckdb` with ChromaDB to avoid SQLite compatibility issues on Streamlit Cloud.

---

## 🧩 Architecture Overview

```plaintext
        +-----------------------------+
        |   User Uploads Documents   |
        +-------------+--------------+
                      |
                      v
          +------------------------+
          |   Text Extraction      |
          | (PDF, DOCX, TXT parsers)|
          +------------------------+
                      |
                      v
       +----------------------------------+
       | Vector Store via Chroma + DuckDB|
       +----------------------------------+
                      |
                      v
        +-----------------------------+
        |      LangChain Pipeline     |
        |    (QA, Summarize, Embed)   |
        +-----------------------------+
                      |
                      v
       +-------------------------------------+
       |     LLMs: Gemini / Groq APIs        |
       +-------------------------------------+
                      |
                      v
        +-----------------------------+
        |     Streamlit Frontend UI   |
        +-----------------------------+
```

---

## 📂 Folder Structure (Typical)

```bash
├── App.py
├── utils/
│   ├── document_loader.py
│   ├── knowledge_base.py
│   ├── chat_utils.py
├── chroma_db/
├── requirements.txt
├── .env
├── assets/
│   ├── logo.png
│   ├── dev_photo.jpg
├── README.md
```

---

## 📸 Screenshots *(Optional — add later)*

* Home Interface
* Document Upload Sidebar
* AI Chat Interaction
* Document Summarization
* Language Selection
* Feedback Interface

---

## 📜 License

MIT License © 2025 \[Abhishek]

---

## 🤝 Acknowledgements

Big thanks to these tools and communities:

* 🤗 HuggingFace
* 🔗 LangChain + LangGraph
* 🧠 Google Gemini
* ⚡ Groq API
* 🧾 PyPDF2 + Tika
* 🌍 Google Translate
* 💻 Streamlit Team

---

> Built with ❤️Abhishek Yadav to make enterprise knowledge smarter, faster, and multilingual.

```
