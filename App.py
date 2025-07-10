import os
os.environ["CHROMA_DB_IMPL"] = "duckdb"
import streamlit as st
from dotenv import load_dotenv
from utils.document_loader import extract_text
from utils.knowledge_base import KnowledgeBase
# Placeholder imports for Gemini, Groq, LangChain, LangGraph
# from langchain.llms import Gemini, Groq
# from langgraph import Graph

load_dotenv()

st.set_page_config(page_title="🧠InsightBot:Company Knowledge Chatbot", layout="wide")
st.markdown('''
<div style="display:flex;align-items:center;justify-content:center;margin-bottom:0.7em;">
  <span class="rainbow-title-animated">
    <span class="emoji-spin"></span> <span class="text-gradient">InsightBot.AI:</span><span class="colorful-flicker">Company Knowledge Assistant</span> <span class="emoji-bounce">🤖</span>
  </span>
</div>
<style>
.rainbow-title-animated {
  font-size: 3.5em;
  font-weight: 900;
  letter-spacing: 2.5px;
  padding: 0.18em 0.7em;
  border-radius: 30px;
  border: 4px solid #fff;
  box-shadow: 0 6px 32px 0 rgba(255,81,47,0.18), 0 2px 16px 0 #1fa2ff80;
  display: inline-block;
  background: linear-gradient(270deg, #ff512f, #f9d423, #1fa2ff, #12d8fa, #a6ffcb, #ff512f, #dd2476, #f9d423, #1fa2ff, #12d8fa, #a6ffcb, #ff512f, #dd2476);
  background-size: 400% 400%;
  animation: rainbowMove 8s linear infinite, popIn 1.2s cubic-bezier(.68,-0.55,.27,1.55) 1;
  -webkit-background-clip: text;
  -webkit-text-fill-color: transparent;
  text-shadow: 0 2px 12px #12d8fa80, 0 1px 0 #fff, 0 0 18px #ff512f80;
  transition: box-shadow 0.3s, border 0.3s;
  filter: drop-shadow(0 2px 16px #1fa2ff80);
  cursor: pointer;
  position: relative;
}
.text-gradient {
  background: linear-gradient(90deg, #ff512f, #f9d423, #1fa2ff, #12d8fa, #a6ffcb, #ff512f, #dd2476);
  background-size: 300% 300%;
  animation: textGradientMove 5s ease-in-out infinite;
  -webkit-background-clip: text;
  -webkit-text-fill-color: transparent;
  margin: 0 0.15em;
}
.colorful-flicker {
  color: #fff;
  background: linear-gradient(90deg, #ff512f, #f9d423, #1fa2ff, #12d8fa, #a6ffcb, #ff512f, #dd2476);
  background-size: 300% 300%;
  -webkit-background-clip: text;
  -webkit-text-fill-color: transparent;
  animation: flicker 2.2s infinite alternate, textGradientMove 6s linear infinite;
  margin: 0 0.15em;
}
.emoji-spin {
  display: inline-block;
  animation: spin 2.5s linear infinite;
}
.emoji-bounce {
  display: inline-block;
  animation: bounce 1.2s infinite alternate;
}
.rainbow-title-animated:hover {
  box-shadow: 0 8px 40px 0 #ff512f99, 0 2px 24px 0 #1fa2ff99;
  border: 4px solid #ff512f;
  animation-play-state: paused;
}
.rainbow-title-animated::after {
  content: '';
  position: absolute;
  left: 10%;
  top: 80%;
  width: 80%;
  height: 8px;
  background: radial-gradient(circle, #fff7 0%, #ff512f33 80%, transparent 100%);
  border-radius: 50%;
  filter: blur(2px);
  opacity: 0.7;
  z-index: -1;
  animation: shadowPulse 2.5s ease-in-out infinite;
}
@keyframes rainbowMove {
  0% {background-position: 0% 50%}
  50% {background-position: 100% 50%}
  100% {background-position: 0% 50%}
}
@keyframes textGradientMove {
  0% {background-position: 0% 50%}
  50% {background-position: 100% 50%}
  100% {background-position: 0% 50%}
}
@keyframes popIn {
  0% {transform: scale(0.7) rotate(-8deg); opacity: 0;}
  60% {transform: scale(1.1) rotate(3deg); opacity: 1;}
  100% {transform: scale(1) rotate(0deg);}
}
@keyframes shadowPulse {
  0%, 100% {opacity: 0.7; transform: scaleX(1);}
  50% {opacity: 1; transform: scaleX(1.15);}
}
@keyframes flicker {
  0%, 19%, 21%, 23%, 25%, 54%, 56%, 100% {
    opacity: 1;
    filter: brightness(1.1) drop-shadow(0 0 8px #fff) drop-shadow(0 0 18px #ff512f);
  }
  20%, 22%, 24%, 55% {
    opacity: 0.7;
    filter: brightness(1.5) drop-shadow(0 0 18px #f9d423) drop-shadow(0 0 24px #1fa2ff);
  }
}
@keyframes spin {
  0% {transform: rotate(0deg);}
  100% {transform: rotate(360deg);}
}
@keyframes bounce {
  0% {transform: translateY(0);}
  100% {transform: translateY(-18px);}
}
</style>
''', unsafe_allow_html=True)

if "kb" not in st.session_state:
    st.session_state.kb = KnowledgeBase()
    st.session_state.docs = []

# --- App Logo at the Top of Sidebar ---
AI_path = "AI.png"  # Ensure this file is in the same directory as your script
try:
    st.sidebar.image(AI_path, use_container_width=True)
except Exception:
    st.sidebar.warning("AI.png file not found. Please check the file path.")

st.sidebar.header("Document Management")
uploaded_files = st.sidebar.file_uploader(
    "Upload documents (PDF, DOCX, TXT)",
    type=["pdf", "docx", "txt"],
    accept_multiple_files=True
)

# --- Multilingual Support ---
try:
    from googletrans import Translator # type: ignore
    translator = Translator()
    st.sidebar.subheader("Language Settings")
    language_options = {
        "English": "en",
        "Hindi": "hi",
        "French": "fr",
        "German": "de",
        "Spanish": "es",
        "Chinese": "zh-cn",
        "Arabic": "ar"
    }
    selected_language = st.sidebar.selectbox("Select your preferred language:", list(language_options.keys()), key="lang_select")
    language_code = language_options[selected_language]
    def translate_text(text, dest_lang):
        if dest_lang == "en":
            return text
        try:
            return translator.translate(text, dest=dest_lang).text
        except Exception:
            return text
except ImportError:
    st.sidebar.warning("Install googletrans for multilingual support: pip install googletrans==4.0.0-rc1")
    language_code = "en"
    def translate_text(text, dest_lang):
        return text

if uploaded_files:
    st.session_state.docs.clear()  # Clear previous docs on new upload
    for uploaded_file in uploaded_files:
        file_path = os.path.join("temp", uploaded_file.name)
        os.makedirs("temp", exist_ok=True)
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        text = extract_text(file_path)
        st.session_state.docs.append({"content": text, "source": uploaded_file.name})
        # Show a preview of the extracted text to the user (translated)
        with st.expander(f"Preview extracted text from {uploaded_file.name}"):
            st.write(translate_text(text[:2000] + ("..." if len(text) > 2000 else ""), language_code))
    st.success(f"Uploaded {len(uploaded_files)} document(s) and extracted text.")

if st.button("Build Knowledge Base"):
    if st.session_state.docs:
        st.session_state.kb.build(st.session_state.docs)
        st.session_state.kb.save()
        st.success("Knowledge base built and saved!")
    else:
        st.warning("Please upload documents first.")

# --- Chat File Uploads & Inline Document Q&A ---
st.subheader("Upload a document for instant Q&A (Chat Upload)")
chat_uploaded_file = st.file_uploader(
    "Upload a document here (PDF, DOCX, TXT) to ask about it directly:",
    type=["pdf", "docx", "txt"],
    key="chat_file_uploader"
)
chat_file_text = None
chat_file_source = None
if chat_uploaded_file:
    chat_file_path = os.path.join("temp", f"chat_{chat_uploaded_file.name}")
    os.makedirs("temp", exist_ok=True)
    with open(chat_file_path, "wb") as f:
        f.write(chat_uploaded_file.getbuffer())
    chat_file_text = extract_text(chat_file_path)
    chat_file_source = chat_uploaded_file.name
    with st.expander(f"Preview extracted text from {chat_uploaded_file.name} (Chat Upload)"):
        st.write(translate_text(chat_file_text, language_code))
    st.info(f"Uploaded {chat_uploaded_file.name} for instant Q&A. Your next question will use this document as context.")

# Conversation history in session state
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []  # Each item: {"role": "user"/"ai", "content": str}

st.header("Chat with your Knowledge Base")
user_message = st.chat_input("Type your message and press Enter...")

# --- Document Search & Filtering ---
st.sidebar.subheader("Search Uploaded Documents")
doc_search_query = st.sidebar.text_input("Search by name, type, or keyword:")
filtered_docs = st.session_state.docs
if doc_search_query:
    doc_search_query_lower = doc_search_query.lower()
    filtered_docs = [
        doc for doc in st.session_state.docs
        if doc_search_query_lower in doc["source"].lower() or doc_search_query_lower in doc["content"].lower()
    ]
    st.sidebar.info(f"Found {len(filtered_docs)} matching document(s).")
if filtered_docs:
    for doc in filtered_docs:
        with st.sidebar.expander(f"{doc['source']}"):
            st.write(translate_text(doc["content"][:500] + ("..." if len(doc["content"]) > 500 else ""), language_code))

# --- Conversation Export & Sharing ---
st.sidebar.subheader("Export & Share Chat")
export_format = st.sidebar.selectbox("Export chat as:", ["Markdown", "Plain Text"], key="export_format")
if st.sidebar.button("Export Chat History"):
    chat_export = ""
    for msg in st.session_state.chat_history:
        role = "You" if msg["role"] == "user" else "AI"
        chat_export += f"**{role}:** {msg['content']}\n\n" if export_format == "Markdown" else f"{role}: {msg['content']}\n\n"
    st.sidebar.download_button(
        label="Download Chat",
        data=chat_export,
        file_name=f"chat_history.{ 'md' if export_format == 'Markdown' else 'txt' }",
        mime="text/markdown" if export_format == "Markdown" else "text/plain"
    )

# --- User Feedback on Answers ---
from collections import defaultdict
if "ai_feedback" not in st.session_state:
    st.session_state.ai_feedback = defaultdict(list)  # key: chat idx, value: list of feedback dicts

def render_ai_message_with_citations_and_feedback(answer, unique_docs, chat_idx):
    import re
    # Find all [Source N] tags
    citation_pattern = r"\\[Source (\\d+)\\]"
    matches = list(re.finditer(citation_pattern, answer))
    last_idx = 0
    expanded = st.session_state.get("expanded_citations", set())
    if "expanded_citations" not in st.session_state:
        st.session_state["expanded_citations"] = set()
    for match in matches:
        start, end = match.span()
        st.markdown(answer[last_idx:start], unsafe_allow_html=True)
        source_num = int(match.group(1))
        # Render clickable citation
        btn_key = f"expand_citation_{source_num}_{chat_idx}"
        if st.button(f"[Source {source_num}]", key=btn_key):
            st.session_state["expanded_citations"].add(btn_key)
        st.markdown(answer[start:end], unsafe_allow_html=True)
        # If expanded, show the full chunk
        if btn_key in st.session_state["expanded_citations"] and source_num-1 < len(unique_docs):
            st.info(f"[Source {source_num}] Full context:\n\n" + unique_docs[source_num-1].page_content)
        last_idx = end
    st.markdown(answer[last_idx:], unsafe_allow_html=True)
    # --- Feedback UI ---
    feedback_key = f"feedback_{chat_idx}"
    col1, col2, col3 = st.columns([1,1,4])
    with col1:
        if st.button("👍", key=f"thumbs_up_{chat_idx}"):
            st.session_state.ai_feedback[chat_idx].append({"feedback": "up"})
    with col2:
        if st.button("👎", key=f"thumbs_down_{chat_idx}"):
            st.session_state.ai_feedback[chat_idx].append({"feedback": "down"})
    with col3:
        user_comment = st.text_input("Comment (optional)", key=f"comment_{chat_idx}")
        if st.button("Submit Comment", key=f"submit_comment_{chat_idx}") and user_comment:
            st.session_state.ai_feedback[chat_idx].append({"feedback": "comment", "text": user_comment})
    # Show feedback summary
    feedbacks = st.session_state.ai_feedback[chat_idx]
    if feedbacks:
        up = sum(1 for f in feedbacks if f["feedback"] == "up")
        down = sum(1 for f in feedbacks if f["feedback"] == "down")
        comments = [f["text"] for f in feedbacks if f["feedback"] == "comment"]
        st.caption(f"Feedback: 👍 {up} | 👎 {down}")
        for c in comments:
            st.caption(f"💬 {c}")

# --- Conversation Topic Tracking & Summarization ---
def extract_topics_from_history(chat_history):
    # Simple keyword extraction from user turns
    from sklearn.feature_extraction.text import CountVectorizer
    user_msgs = [msg['content'] for msg in chat_history if msg['role'] == 'user']
    if not user_msgs:
        return []
    vectorizer = CountVectorizer(stop_words='english', max_features=10)
    X = vectorizer.fit_transform(user_msgs)
    topics = vectorizer.get_feature_names_out()
    return topics

def summarize_history(chat_history, max_tokens=1200):
    # Summarize chat history if too long for LLM context
    all_text = '\n'.join([
        f"User: {msg['content']}" if msg['role']=='user' else f"AI: {msg['content']}"
        for msg in chat_history
    ])
    # Use tiktoken to count tokens (or fallback to len)
    try:
        import tiktoken
        enc = tiktoken.get_encoding('cl100k_base')
        tokens = len(enc.encode(all_text))
    except Exception:
        tokens = len(all_text.split())
    if tokens <= max_tokens:
        return all_text
    # If too long, summarize
    summary = all_text[-max_tokens*5:]
    return summary

# --- Sidebar: Topics & Controls ---
st.sidebar.subheader("Conversation Topics")
topics = extract_topics_from_history(st.session_state.chat_history)
if len(topics) > 0:
    st.sidebar.write(", ".join(topics))
else:
    st.sidebar.caption("No topics detected yet.")

if st.sidebar.button("Clear Chat History"):
    st.session_state.chat_history = []
    st.rerun()

# --- Display running summary ---
st.sidebar.subheader("Conversation Summary")
if st.session_state.chat_history:
    summary = summarize_history(st.session_state.chat_history, max_tokens=200)
    st.sidebar.caption(summary[:300] + ("..." if len(summary) > 300 else ""))
    with st.sidebar.expander("Preview Full Chat History", expanded=False):
        chat_export = ""
        for msg in st.session_state.chat_history:
            role = "You" if msg["role"] == "user" else "AI"
            chat_export += f"**{role}:** {msg['content']}\n\n"
        st.markdown(chat_export, unsafe_allow_html=True)
        st.info("Scroll to view the full chat. You can copy and save this history.")
else:
    st.sidebar.caption("No conversation yet.")

# --- Persona Selection ---
st.sidebar.subheader("Chat Persona")
persona = st.sidebar.selectbox(
    "Choose AI style:",
    ["Helpful", "Formal", "Friendly", "Concise"],
    key="persona_select"
)
persona_instructions = {
    "Helpful": "You are a helpful assistant.",
    "Formal": "You are a formal, professional assistant. Use formal language.",
    "Friendly": "You are a friendly, casual assistant. Use friendly, approachable language.",
    "Concise": "You are a concise assistant. Keep answers short and to the point."
}

# --- Developer Info and Image ---
st.sidebar.markdown("👨‍💻Developer:- Abhishek💖Yadav")
developer_path = "pic.jpg"  # Ensure this file is in the same directory as your script
try:
    st.sidebar.image(developer_path, use_container_width=True)
except Exception:
    st.sidebar.warning("pic.jpg file not found. Please check the file path.")

# --- Pin/Starred Answers ---
if "starred_answers" not in st.session_state:
    st.session_state.starred_answers = set()

# --- Enhanced Custom CSS for Vibrant, Modern, and Interactive UI ---
custom_css = '''
<style>
body, .stApp {
    background: linear-gradient(135deg, #232526 0%, #1fa2ff 100%) !important;
    color: #f0f0f0 !important;
}
header, .st-emotion-cache-1avcm0n, .st-emotion-cache-1dp5vir, .st-emotion-cache-1v0mbdj {
    background: transparent !important;
}
.st-emotion-cache-1v0mbdj, .st-emotion-cache-1dp5vir {
    border-radius: 22px !important;
    box-shadow: 0 6px 32px 0 rgba(31,162,255,0.18);
    background: linear-gradient(135deg, #232526 0%, #1fa2ff 100%) !important;
}
.stButton>button {
    background: linear-gradient(90deg, #ff512f 0%, #dd2476 50%, #1fa2ff 100%) !important;
    color: #fff !important;
    border: none !important;
    border-radius: 12px !important;
    padding: 0.7em 2em !important;
    font-weight: bold;
    margin: 0.3em 0.7em 0.3em 0;
    box-shadow: 0 3px 12px 0 rgba(31,162,255,0.18);
    transition: 0.2s;
    font-size: 1.1em;
}
.stButton>button:hover {
    background: linear-gradient(90deg, #1fa2ff 0%, #dd2476 50%, #ff512f 100%) !important;
    color: #fff !important;
    box-shadow: 0 6px 24px 0 rgba(255,81,47,0.25);
}
.stTextInput>div>input, .stTextArea>div>textarea {
    background: #232526 !important;
    color: #fff !important;
    border: 2px solid #1fa2ff !important;
    border-radius: 10px !important;
    font-size: 1.1em;
}
.stSidebar, .st-emotion-cache-1v0mbdj {
    background: linear-gradient(135deg, #18191a 0%, #1fa2ff 100%) !important;
}
.st-emotion-cache-1dp5vir {
    background: #232526 !important;
}
.stMarkdown, .stMarkdown p {
    color: #f0f0f0 !important;
    font-size: 1.1em;
}
.stExpanderHeader {
    color: #ff512f !important;
    font-weight: bold;
    font-size: 1.15em;
}
.stExpander {
    background: linear-gradient(90deg, #232526 0%, #1fa2ff 100%) !important;
    border-radius: 14px !important;
    border: 2px solid #1fa2ff !important;
}
.stAlert, .stInfo, .stSuccess, .stWarning, .stError {
    border-radius: 14px !important;
    font-size: 1.08em;
}
.st-emotion-cache-1v0mbdj .st-bb, .st-emotion-cache-1dp5vir .st-bb {
    background: transparent !important;
}
::-webkit-scrollbar {
    width: 10px;
    background: #232526;
}
::-webkit-scrollbar-thumb {
    background: linear-gradient(135deg, #1fa2ff 0%, #ff512f 100%);
    border-radius: 8px;
}
.stChatInputContainer, .stChatInputContainer textarea {
    background: #18191a !important;
    color: #fff !important;
    border-radius: 10px !important;
    font-size: 1.1em;
}
</style>
'''
st.markdown(custom_css, unsafe_allow_html=True)

# --- Enhanced App Title with Animated Gradient and Icon ---
st.markdown('''
<div style="display:flex;align-items:center;justify-content:center;margin-bottom:0.5em;">
    <span style="font-size:3em;font-weight:900;background:linear-gradient(270deg,#ff512f,#dd2476,#1fa2ff,#12d8fa,#a6ffcb,#ff512f);background-size:1200% 1200%;animation:gradientMove 8s ease infinite;-webkit-background-clip:text;-webkit-text-fill-color:transparent;">🤖 Interactive Chatbot: <span style='font-size:0.5em;font-weight:600;'>For Company Docs and Policies</span></span>
</div>
<style>
@keyframes gradientMove {
    0%{background-position:0% 50%}
    50%{background-position:100% 50%}
    100%{background-position:0% 50%}
}
</style>
''', unsafe_allow_html=True)

# --- Enhanced Chat Bubble Styles for Vibrant UI ---
user_bubble = "background: linear-gradient(90deg,#1fa2ff,#12d8fa,#a6ffcb); color:#232526; padding:16px; border-radius:18px; margin-bottom:10px; font-weight:600; font-size:1.13em; box-shadow:0 2px 12px 0 rgba(31,162,255,0.10);"
ai_bubble = "background: linear-gradient(90deg,#ff512f,#dd2476); color:#fff; padding:16px; border-radius:18px; margin-bottom:10px; font-weight:600; font-size:1.13em; box-shadow:0 2px 12px 0 rgba(221,36,118,0.10);"

# --- AI-Powered Document Summarization & Section Navigator ---
from typing import List, Dict

def get_document_summary(text: str) -> str:
    """Generate a concise summary for a document using Gemini API."""
    try:
        from langchain_google_genai import ChatGoogleGenerativeAI
        gemini_api_key = os.environ.get("GEMINI_API_KEY")
        if not gemini_api_key:
            return "[Gemini API key not set. Cannot summarize.]"
        llm = ChatGoogleGenerativeAI(api_key=gemini_api_key, model="models/gemini-1.5-flash-latest")
        prompt = (
            "Summarize the following document in 5-7 concise bullet points, focusing on the most important information, policies, or topics. "
            "Do not include generic statements.\n\nDocument:\n" + text[:6000]
        )
        summary = llm.invoke(prompt)
        return summary.content if hasattr(summary, 'content') else str(summary)
    except Exception as e:
        return f"[Error summarizing: {e}]"

def extract_sections(text: str) -> List[Dict[str, str]]:
    """Extract key sections/topics from a document using simple heuristics or LLM."""
    import re
    # Heuristic: Look for lines that look like section headers (all caps, numbers, or >20 chars)
    lines = text.splitlines()
    sections = []
    for i, line in enumerate(lines):
        if (len(line) > 18 and line.isupper()) or re.match(r"^\d+\. ", line):
            start = max(0, i-1)
            end = min(len(lines), i+8)
            section_text = "\n".join(lines[start:end])
            sections.append({"title": line.strip(), "content": section_text.strip()})
    # Fallback: If no sections found, just return first 2 chunks
    if not sections:
        chunk_size = max(400, len(text)//2)
        sections = [
            {"title": "Start of Document", "content": text[:chunk_size]},
            {"title": "Middle of Document", "content": text[chunk_size:chunk_size*2]}
        ]
    return sections[:8]

# --- Sidebar: Document Summaries & Section Navigator ---
if st.session_state.docs:
    st.sidebar.subheader("📄 Document Summaries & Navigator")
    for doc in st.session_state.docs:
        doc_name = doc["source"]
        doc_text = doc["content"]
        # Cache summaries and sections in session_state
        if "summaries" not in st.session_state:
            st.session_state.summaries = {}
        if "sections" not in st.session_state:
            st.session_state.sections = {}
        if doc_name not in st.session_state.summaries:
            with st.spinner(f"Summarizing {doc_name}..."):
                st.session_state.summaries[doc_name] = get_document_summary(doc_text)
        if doc_name not in st.session_state.sections:
            st.session_state.sections[doc_name] = extract_sections(doc_text)
        with st.sidebar.expander(f"{doc_name} - Summary", expanded=False):
            st.markdown(st.session_state.summaries[doc_name])
        with st.sidebar.expander(f"{doc_name} - Section Navigator", expanded=False):
            for i, section in enumerate(st.session_state.sections[doc_name]):
                if st.button(f"Jump to: {section['title'][:40]}", key=f"jump_{doc_name}_{i}"):
                    st.session_state["section_preview"] = section["content"]
            if st.session_state.get("section_preview"):
                st.info(st.session_state["section_preview"])

# --- AI-Driven Action Suggestions & Smart Shortcuts ---
def get_action_suggestions(chat_history: List[Dict], docs: List[Dict]) -> List[str]:
    """Suggest follow-up questions or actions based on chat context."""
    try:
        from langchain_google_genai import ChatGoogleGenerativeAI
        gemini_api_key = os.environ.get("GEMINI_API_KEY")
        if not gemini_api_key:
            return []
        llm = ChatGoogleGenerativeAI(api_key=gemini_api_key, model="models/gemini-1.5-flash-latest")
        last_user = next((msg["content"] for msg in reversed(chat_history) if msg["role"]=="user"), "")
        doc_names = ", ".join([d["source"] for d in docs])
        prompt = (
            "Given the user's last question and the available documents (" + doc_names + "), "
            "suggest 3-4 smart follow-up questions or actions the user might want to take next. "
            "Be specific and context-aware. Return each suggestion as a separate line.\n\n"
            f"Last user message: {last_user}\n"
        )
        suggestions = llm.invoke(prompt)
        lines = suggestions.content.splitlines() if hasattr(suggestions, 'content') else str(suggestions).splitlines()
        return [l.strip('-• ') for l in lines if l.strip()]
    except Exception:
        return []

# --- After AI response, show action suggestions & smart shortcuts ---
if st.session_state.chat_history and st.session_state.chat_history[-1]["role"] == "ai":
    suggestions = get_action_suggestions(st.session_state.chat_history, st.session_state.docs)
    if suggestions:
        st.markdown("""
        <div style='margin-top:1.5em; margin-bottom:0.5em;'>
            <b style='font-size:1.18em;'>💡 Smart Suggestions:</b>
        </div>
        """, unsafe_allow_html=True)
        if "suggestion_history" not in st.session_state:
            st.session_state.suggestion_history = []
        if "pinned_suggestions" not in st.session_state:
            st.session_state.pinned_suggestions = set()
        for i, suggestion in enumerate(suggestions):
            with st.container():
                col1, col2, col3, col4 = st.columns([6,1,1,1])
                with col1:
                    if f"edit_suggestion_{i}" not in st.session_state:
                        st.session_state[f"edit_suggestion_{i}"] = suggestion
                    edited = st.text_input("", value=st.session_state[f"edit_suggestion_{i}"], key=f"edit_suggestion_input_{i}", label_visibility="collapsed")
                    st.session_state[f"edit_suggestion_{i}"] = edited
                with col2:
                    ask_btn = st.button("🚀 Ask", key=f"ask_suggestion_{i}", help="Send this suggestion as your next question.")
                    if ask_btn:
                        st.session_state.chat_history.append({"role": "user", "content": edited})
                        st.session_state.suggestion_history.append(edited)
                        st.rerun()
                with col3:
                    preview_btn = st.button("👁️ Preview", key=f"preview_suggestion_{i}", help="Preview the AI's answer for this suggestion.")
                    if preview_btn:
                        st.session_state.chat_history.append({"role": "user", "content": edited})
                        st.session_state["preview_mode"] = True
                        st.rerun()
                with col4:
                    pin_btn = st.button("📌", key=f"pin_suggestion_{i}", help="Pin this suggestion for later.")
                    if pin_btn:
                        st.session_state.pinned_suggestions.add(edited)
                        st.toast("Pinned!", icon="📌")
                # Copy button (fix: escape single quotes outside f-string)
                edited_js = edited.replace("'", "\\'")
                st.markdown(f'<button style="margin-top:0.3em; background:linear-gradient(90deg,#1fa2ff,#12d8fa,#a6ffcb); color:#232526; border:none; border-radius:8px; padding:0.3em 1.2em; font-weight:600; font-size:1em; cursor:pointer; transition:0.15s;" onclick="navigator.clipboard.writeText(\'{edited_js}\')">📋 Copy</button>', unsafe_allow_html=True)
        # Quick Ask All button
        if st.button("✨ Quick Ask All", key="quick_ask_all", help="Send all suggestions as separate questions."):
            for i, suggestion in enumerate(suggestions):
                edited = st.session_state.get(f"edit_suggestion_{i}", suggestion)
                st.session_state.chat_history.append({"role": "user", "content": edited})
                st.session_state.suggestion_history.append(edited)
            st.rerun()
        # Show pinned suggestions
        if st.session_state.pinned_suggestions:
            st.markdown("<div style='margin-top:0.7em;'><b>📌 Pinned Suggestions:</b></div>", unsafe_allow_html=True)
            for pin in st.session_state.pinned_suggestions:
                st.markdown(f'<div style="background:linear-gradient(90deg,#ff512f,#dd2476,#1fa2ff); color:#fff; border-radius:10px; padding:0.5em 1em; margin-bottom:0.3em; font-weight:500;">{pin}</div>', unsafe_allow_html=True)
        # Show suggestion history
        if st.session_state.suggestion_history:
            st.markdown("<div style='margin-top:0.7em;'><b>🕑 Used Suggestions:</b></div>", unsafe_allow_html=True)
            for hist in st.session_state.suggestion_history[-5:][::-1]:
                st.markdown(f'<div style="background:linear-gradient(90deg,#232526,#1fa2ff); color:#fff; border-radius:10px; padding:0.4em 1em; margin-bottom:0.2em; font-size:0.98em;">{hist}</div>', unsafe_allow_html=True)
        # Add subtle CSS for cards and hover
        st.markdown('''
        <style>
        .stTextInput>div>input:focus {
            border: 2.5px solid #1fa2ff !important;
            box-shadow: 0 0 8px #1fa2ff55;
        }
        .stButton>button[title*="Ask"] {
            background: linear-gradient(90deg,#1fa2ff,#12d8fa,#a6ffcb) !important;
            color: #232526 !important;
            border-radius: 10px !important;
            font-weight: bold;
            font-size: 1.08em;
            box-shadow: 0 2px 8px 0 rgba(31,162,255,0.10);
            transition: 0.18s;
        }
        .stButton>button[title*="Ask"]:hover {
            background: linear-gradient(90deg,#ff512f,#dd2476) !important;
            color: #fff !important;
            box-shadow: 0 4px 16px 0 rgba(221,36,118,0.13);
        }
        .stButton>button[title*="Quick Ask All"] {
            background: linear-gradient(90deg,#ff512f,#dd2476,#1fa2ff) !important;
            color: #fff !important;
            border-radius: 16px !important;
            font-weight: bold;
            font-size: 1.13em;
            margin-top: 0.7em;
            box-shadow: 0 2px 12px 0 rgba(31,162,255,0.13);
        }
        .stButton>button[title*="Quick Ask All"]:hover {
            background: linear-gradient(90deg,#1fa2ff,#dd2476,#ff512f) !important;
            color: #fff !important;
        }
        </style>
        ''', unsafe_allow_html=True)

# --- Display chat history with edit/delete/pin ---
# Ensure unique_docs is always defined to avoid NameError
if 'unique_docs' not in locals():
    unique_docs = []

for idx, msg in enumerate(st.session_state.chat_history):
    if msg["role"] == "user":
        col1, col2, col3 = st.columns([8,1,1])
        with col1:
            st.markdown(f'<div style="{user_bubble}"><b>You:</b> {translate_text(msg["content"], language_code)}</div>', unsafe_allow_html=True)
        with col2:
            if st.button("✏️", key=f"edit_{idx}"):
                new_text = st.text_input("Edit your message:", value=msg["content"], key=f"edit_input_{idx}")
                if st.button("Save Edit", key=f"save_edit_{idx}"):
                    st.session_state.chat_history[idx]["content"] = new_text
                    st.rerun()
        with col3:
            if st.button("🗑️", key=f"delete_{idx}"):
                st.session_state.chat_history.pop(idx)
                st.rerun()
    else:
        col1, col2, col3 = st.columns([7,1,1])
        with col1:
            st.markdown(f'<div style="{ai_bubble}"><b>AI:</b> {translate_text(msg["content"], language_code)}</div>', unsafe_allow_html=True)
            render_ai_message_with_citations_and_feedback(translate_text(msg["content"], language_code), unique_docs if idx == len(st.session_state.chat_history)-1 and 'unique_docs' in locals() else [], idx)
        with col2:
            if st.button("⭐", key=f"star_{idx}"):
                st.session_state.starred_answers.add(idx)
        with col3:
            if st.button("🔄", key=f"regen_{idx}"):
                user_msg = st.session_state.chat_history[idx-1]["content"] if idx > 0 else ""
                st.session_state.chat_history = st.session_state.chat_history[:idx]
                st.rerun()
        if idx in st.session_state.starred_answers:
            st.caption("⭐ Starred")

# --- Typing Indicator ---
typing_placeholder = st.empty()

if user_message:
    # Translate user message to English for LLM if needed
    user_message_en = user_message if language_code == "en" else translate_text(user_message, "en")
    # Add user message to history
    st.session_state.chat_history.append({"role": "user", "content": user_message})
    st.session_state.kb.load()
    # Retrieve relevant chunks for the latest user message
    docs = []
    if chat_file_text:
        from langchain.docstore.document import Document
        splits = st.session_state.kb.text_splitter.split_text(chat_file_text)
        docs = [Document(page_content=split, metadata={"source": chat_file_source}) for split in splits][:8]
    else:
        docs = st.session_state.kb.query(user_message, k=8)
    seen_chunks = set()
    unique_docs = []
    for doc in docs:
        chunk = doc.page_content.strip()
        if chunk not in seen_chunks:
            seen_chunks.add(chunk)
            unique_docs.append(doc)
    # Use full or summarized chat history for context
    chat_context = summarize_history(st.session_state.chat_history, max_tokens=1200)
    doc_context = "\n\n".join([f"[Source {i+1}] {doc.page_content[:600]}" for i, doc in enumerate(unique_docs)])
    prompt = (
        persona_instructions.get(persona, "You are a helpful assistant.") +
        " Use ONLY the provided context from company documents to answer the user's question. "
        "Cite the source number (e.g., [Source 1]) after each fact. "
        "If the answer is not in the context, say 'I don't know.'\n\n"
        f"Conversation so far:\n{chat_context}\n\nContext:\n{doc_context}\n\nUser: {user_message_en}\nAI (with citations):"
    )
    typing_placeholder.info("AI is typing...")
    try:
        from langchain_google_genai import ChatGoogleGenerativeAI
        gemini_api_key = os.environ.get("GEMINI_API_KEY")
        if not gemini_api_key:
            st.error("GEMINI_API_KEY is not set. Please check your .env file and restart the app.")
        else:
            with st.spinner("Generating answer..."):
                llm = ChatGoogleGenerativeAI(api_key=gemini_api_key, model="models/gemini-1.5-flash-latest")
                ai_answer = llm.invoke(prompt)
            if hasattr(ai_answer, 'content'):
                answer_text = ai_answer.content
            else:
                answer_text = str(ai_answer)
            # Translate AI answer to user language if needed
            answer_text_translated = answer_text if language_code == "en" else translate_text(answer_text, language_code)
            st.session_state.chat_history.append({"role": "ai", "content": answer_text_translated})
            render_ai_message_with_citations_and_feedback(answer_text_translated, unique_docs, len(st.session_state.chat_history)-1)
    except Exception as e:
        st.error(f"Error generating AI answer: {e}")
    typing_placeholder.empty()

st.sidebar.markdown("---")
st.sidebar.write("Built💖with Streamlit, LangChain,💖LangGraph,Groq")

# --- Enhanced Chat Input Box and Send Button Styling ---
custom_css2 = '''
<style>
.stChatInputContainer, .stChatInputContainer textarea {
    background: linear-gradient(90deg, #232526 0%, #1fa2ff 100%) !important;
    color: #fff !important;
    border-radius: 16px !important;
    font-size: 1.15em !important;
    border: 2.5px solid #ff512f !important;
    box-shadow: 0 2px 12px 0 rgba(31,162,255,0.13);
    margin-bottom: 0.5em !important;
}
.stChatInputContainer textarea:focus {
    border: 2.5px solid #dd2476 !important;
    outline: none !important;
}
.stChatInputContainer button {
    background: linear-gradient(90deg, #ff512f 0%, #dd2476 50%, #1fa2ff 100%) !important;
    color: #fff !important;
    border: none !important;
    border-radius: 50% !important;
    width: 48px !important;
    height: 48px !important;
    font-size: 1.7em !important;
    margin-left: 0.5em !important;
    box-shadow: 0 3px 12px 0 rgba(31,162,255,0.18);
    display: flex; align-items: center; justify-content: center;
    transition: 0.2s;
}
.stChatInputContainer button:hover {
    background: linear-gradient(90deg, #1fa2ff 0%, #dd2476 50%, #ff512f 100%) !important;
    color: #fff !important;
    box-shadow: 0 6px 24px 0 rgba(255,81,47,0.25);
}
.stChatInputContainer svg {
    color: #fff !important;
    font-size: 1.5em !important;
}
</style>
'''
st.markdown(custom_css2, unsafe_allow_html=True)
