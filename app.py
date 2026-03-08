import os
import logging
from typing import List
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

# LangChain
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.retrievers import BM25Retriever
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate

# ---------------- CONFIG ---------------- #

PDF_FILES = [
    "NEP_Final_English.pdf",
    "ABSS_2025_Concept_Note.pdf"
]

FAISS_INDEX_DIR = "faiss_index"

EMBEDDING_MODEL = "all-MiniLM-L6-v2"
LLM_MODEL = "llama-3.3-70b-versatile"

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ---------------- FASTAPI ---------------- #

app = FastAPI(title="NEP 2020 Policy AI Assistant")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class Query(BaseModel):
    question: str

# ---------------- RAG SYSTEM ---------------- #

class EducationPolicyRAG:

    def __init__(self):

        self.embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
        self.vectorstore = None
        self.bm25 = None
        self.llm = ChatGroq(
            model_name=LLM_MODEL,
            temperature=0
        )

    # ---------------- LOAD DOCUMENTS ---------------- #

    def load_documents(self):

        documents = []

        for pdf in PDF_FILES:

            if not os.path.exists(pdf):
                raise FileNotFoundError(f"{pdf} not found")

            logger.info(f"Loading {pdf} ...")

            loader = PyPDFLoader(pdf)
            docs = loader.load()

            for doc in docs:
                doc.metadata["source"] = pdf

            documents.extend(docs)

        logger.info(f"Loaded {len(documents)} documents.")

        return documents

    # ---------------- BUILD INDEX ---------------- #

    def build_index(self):

        logger.info("Building FAISS index...")

        documents = self.load_documents()

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=150
        )

        split_docs = splitter.split_documents(documents)

        self.vectorstore = FAISS.from_documents(
            split_docs,
            self.embeddings
        )

        self.vectorstore.save_local(FAISS_INDEX_DIR)

        self.bm25 = BM25Retriever.from_documents(split_docs)

        logger.info("Index built successfully.")

    # ---------------- LOAD INDEX ---------------- #

    def load_index(self):

        logger.info("Loading FAISS index...")

        self.vectorstore = FAISS.load_local(
            FAISS_INDEX_DIR,
            self.embeddings,
            allow_dangerous_deserialization=True
        )

        docs = list(self.vectorstore.docstore._dict.values())

        self.bm25 = BM25Retriever.from_documents(docs)

        logger.info("Index loaded successfully.")

    # ---------------- INITIALIZE ---------------- #

    def initialize(self):

        if os.path.exists(FAISS_INDEX_DIR):
            self.load_index()
        else:
            self.build_index()

    # ---------------- HYBRID RETRIEVAL ---------------- #

    def hybrid_retrieve(self, query: str, k: int = 5):

        dense_docs = self.vectorstore.similarity_search(query, k=k)

        self.bm25.k = k
        sparse_docs = self.bm25.invoke(query)

        combined = []
        seen = set()

        for doc in dense_docs + sparse_docs:
            if doc.page_content not in seen:
                combined.append(doc)
                seen.add(doc.page_content)

        return combined[:5]

    # ---------------- ASK ---------------- #

    def ask(self, question: str):

        docs = self.hybrid_retrieve(question)

        context = "\n\n".join([doc.page_content for doc in docs])

        template = """
You are an expert assistant for Indian Education Policy.

Answer questions using the NEP 2020 and ABSS 2025 documents.

Guidelines:
- Provide clear explanations.
- If possible mention the policy concept.
- Keep answers structured and concise.

Context:
{context}

Question:
{question}

Answer:
"""

        prompt = ChatPromptTemplate.from_template(template)

        chain = prompt | self.llm

        response = chain.invoke({
            "context": context,
            "question": question
        })

        return response.content


# ---------------- INITIALIZE ---------------- #

rag = EducationPolicyRAG()

@app.on_event("startup")
def startup_event():

    logger.info("Initializing NEP Policy RAG...")

    rag.initialize()

    logger.info("System Ready.")

# ---------------- ROUTES ---------------- #

@app.get("/", response_class=HTMLResponse)
def home():

    if os.path.exists("index.html"):

        with open("index.html", "r", encoding="utf-8") as f:
            return f.read()

    return "<h3>index.html not found</h3>"


@app.post("/chat")
def chat(query: Query):

    if not query.question.strip():
        raise HTTPException(status_code=400, detail="Empty question")

    try:

        answer = rag.ask(query.question)

        return {"answer": answer}

    except Exception as e:

        logger.error(str(e))

        raise HTTPException(
            status_code=500,
            detail="Internal Server Error"
        )