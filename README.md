
---

````markdown
# 🧠 InsightBot.AI: Company Knowledge Assistant 🤖

🔗 **Live App:** [https://insightbot-ai-company-knowledge-assistant.streamlit.app/](https://insightbot-ai-company-knowledge-assistant.streamlit.app/)

**InsightBot.AI** is a cutting-edge, multilingual AI-powered knowledge assistant designed for modern companies. Built with Streamlit, LangChain, Gemini, Groq, and LangGraph, it transforms documents into a powerful, interactive chatbot interface. Whether on your local machine or deployed to the cloud — it just works 🚀

---

## 📌 App Highlights

### 📂 1. Document Upload & Management
- Upload **multiple documents** (`PDF`, `DOCX`, `TXT`) directly from the sidebar.
- Auto-extracts text using:
  - `PyPDF2` (primary),
  - `Tika` (fallback),
  - `docx2txt`,
  - or plain `.txt` reading.
- Document **search**, **preview**, and **filtering** by name/type/keyword.
- Sidebar document viewer for quick management. 🧾

---

### 🧠 2. Knowledge Base Construction
- Creates a **vector-based knowledge base** using:
  - `LangChain`
  - `ChromaDB` with `DuckDB` backend for full Streamlit Cloud compatibility.
- Embeddings via **HuggingFace** models (forced on CPU).
- Persistent vector store — rebuild or update with ease! 🧱

---

### 💬 3. AI-Powered Chatbot
- Ask questions based on your uploaded content.
- Uses **Gemini LLM** and **Groq API** for lightning-fast, intelligent responses.
- 📚 Answers include **source citations** and expandable full context.
- Maintains **chat history**, exportable as Markdown or plain text.

---

### 📑 4. Document Summarization & Smart Navigation
- Automatically summarizes uploaded docs using **Gemini**.
- Extracts key **sections/topics** with jump-to-preview buttons.
- Great for navigating large files quickly! 🧭

---

### 💡 5. Smart Suggestions & Action Shortcuts
- After every AI answer, get **context-aware follow-up ideas**.
- Edit, copy, pin, or ask all with 1-click 🧠
- History of suggestions stored in elegant card-style UI. 📇

---

### 🌍 6. Multilingual Support
- Chat with InsightBot in your preferred language!
- Supports:
  - English 🇬🇧
  - Hindi 🇮🇳
  - French 🇫🇷
  - German 🇩🇪
  - Spanish 🇪🇸
  - Chinese 🇨🇳
  - Arabic 🇸🇦
- Uses Google Translate API for seamless back-and-forth translation.

---

### ⭐ 7. User Feedback & Chat Management
- React to answers with 👍 / 👎 + comments.
- Edit, delete, regenerate, or star messages ⭐
- Topic extraction and conversation summaries included!

---

### 🎨 8. Modern UI/UX
- Animated, colorful app title 🎉
- Developer profile photo + logo in sidebar.
- Stylish chat interface with responsive design.
- Custom CSS for buttons, input boxes, and chat bubbles 💅

---

### 🛠️ 9. Robust Error Handling
- Gracefully handles:
  - Missing dependencies,
  - Model errors,
  - Extraction issues.
- Friendly messages and fallback logic included! 🙌

---

### ☁️ 10. Cloud Deployment Ready
- ✅ Fully deployable to [Streamlit Cloud](https://streamlit.io/cloud)
- ✅ Uses `DuckDB` to avoid SQLite version issues.
- ✅ All dependencies listed in `requirements.txt` for one-click deployment.

---

## 📦 Tech Stack

| Category              | Tools & APIs Used                           |
|-----------------------|---------------------------------------------|
| Framework             | `Streamlit`                                 |
| LLMs & Agents         | `Gemini` via `langchain-google-genai`, `Groq`, `LangChain`, `LangGraph` |
| File Processing       | `PyPDF2`, `Tika`, `docx2txt`                |
| Vector DB             | `ChromaDB` + `DuckDB` backend               |
| Embeddings            | `HuggingFace` CPU models                    |
| Translation           | `Googletrans` (Multilingual Chat Support)   |
| Deployment            | Streamlit Cloud Compatible                  |

---

## 🚀 Getting Started Locally

```bash
git clone https://github.com/abhishekkumar62000/InsightBot.AI--Company-Knowledge-Assistant.git
cd insightbot-ai
pip install -r requirements.txt
streamlit run App.py
````

---

## 📄 License

MIT License © 2025 \[MIT]

---

## 🙌 Acknowledgements

* HuggingFace 🤗
* Google Gemini AI 🔮
* Groq 🧠⚡
* LangChain & LangGraph 🔗
* Streamlit ❤️

---

> Made with ❤️ by \[Abhishek Yadav] — Empowering companies with smart knowledge access.

```

