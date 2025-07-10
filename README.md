

````markdown
# 🧠 InsightBot.AI: Company Knowledge Assistant 🤖

🔗 **Live App**: [https://insightbot-ai-company-knowledge-assistant.streamlit.app/](https://insightbot-ai-company-knowledge-assistant.streamlit.app/)

InsightBot.AI is a powerful AI-powered knowledge assistant tailored for companies and teams. It allows users to **upload internal documents**, builds a **searchable knowledge base**, and enables smart interaction through **natural language queries** in multiple languages — all through a beautiful, modern Streamlit UI.

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
git clone https://github.com/your-username/insightbot-ai.git
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

MIT License © 2025 \[Your Name]

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

---

✅ Let me know if you'd like this `README.md` saved and sent to you as a downloadable file. I can also help:
- Add GitHub badges
- Create a project logo/banner
- Generate screenshots mockups for your repo

Want that?
```
