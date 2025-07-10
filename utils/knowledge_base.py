import os
from dotenv import load_dotenv
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document

load_dotenv()

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

class KnowledgeBase:
    def __init__(self, persist_dir="vectorstore"):
        self.persist_dir = persist_dir
        self.embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
        self.vectorstore = None
        self.text_splitter = RecursiveCharacterTextSplitter(chunk_size=400, chunk_overlap=80)

    def build(self, docs):
        texts = []
        metadatas = []
        for doc in docs:
            splits = self.text_splitter.split_text(doc["content"])
            for split in splits:
                texts.append(split)
                metadatas.append({"source": doc["source"]})
        self.vectorstore = Chroma.from_texts(texts, self.embeddings, metadatas=metadatas, persist_directory=self.persist_dir)

    def save(self):
        if self.vectorstore:
            self.vectorstore.persist()

    def load(self):
        if os.path.exists(self.persist_dir):
            self.vectorstore = Chroma(persist_directory=self.persist_dir, embedding_function=self.embeddings)

    def query(self, query, k=4):
        if not self.vectorstore:
            return []
        docs = self.vectorstore.similarity_search(query, k=k)
        return docs
