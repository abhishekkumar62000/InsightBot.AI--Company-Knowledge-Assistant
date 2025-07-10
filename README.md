# Custom Company Knowledge Base Chatbot

This Streamlit app allows users to upload company documents (PDF, TXT, DOCX), builds a knowledge base using LangChain, LangGraph, Gemini API, and Groq API, and enables interactive Q&A with source citations, context-aware chat, and document management.

## Features
- Multi-document upload and management
- Retrieval-Augmented Generation (RAG) for Q&A
- Source citation for answers
- Context-aware chat
- Built with Streamlit, Python, LangChain, LangGraph, Gemini API, and Groq API

## Getting Started
1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. Set up your `.env` file with API keys for Gemini and Groq.
3. Run the app:
   ```bash
   streamlit run app.py
   ```

## File Structure
- `app.py`: Main Streamlit app
- `requirements.txt`: Python dependencies
- `.env`: API keys
- `utils/`: Helper functions (document processing, RAG, etc.)

## Notes
- Ensure your API keys are valid and have sufficient quota.
- For best results, upload clear and relevant documents.
