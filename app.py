import os
import pickle
import logging
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

from langchain_community.document_loaders import PyPDFLoader
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

PDF_FILES = [
    "NEP_Final_English.pdf",
    "ABSS_2025_Concept_Note.pdf"
]

CACHE_FILE = "policy_docs.pkl"
LLM_MODEL = "llama-3.3-70b-versatile"

app = FastAPI(title="NEP Policy Assistant")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class Query(BaseModel):
    question: str


class SimplePolicyRAG:

    def __init__(self):
        self.documents = []
        self.llm = ChatGroq(
            model_name=LLM_MODEL,
            temperature=0
        )

    # -------- LOAD OR CACHE DOCUMENTS -------- #

    def load_documents(self):

        if os.path.exists(CACHE_FILE):
            logger.info("Loading documents from cache...")
            with open(CACHE_FILE, "rb") as f:
                self.documents = pickle.load(f)
            return

        logger.info("Parsing PDFs for first time...")

        docs = []

        for pdf in PDF_FILES:

            if not os.path.exists(pdf):
                raise FileNotFoundError(f"{pdf} not found")

            loader = PyPDFLoader(pdf)
            pdf_docs = loader.load()

            for d in pdf_docs:
                docs.append(d.page_content)

        self.documents = docs

        # Save cache
        with open(CACHE_FILE, "wb") as f:
            pickle.dump(self.documents, f)

        logger.info(f"Cached {len(docs)} document pages")

    # -------- SIMPLE SEARCH -------- #

    def search(self, query, k=5):

        query = query.lower()

        scored = []

        for text in self.documents:
            score = text.lower().count(query)
            if score > 0:
                scored.append((score, text))

        scored.sort(reverse=True)

        results = [s[1] for s in scored[:k]]

        if not results:
            results = self.documents[:k]

        return results

    # -------- ASK -------- #

    def ask(self, question):

        docs = self.search(question)

        context = "\n\n".join(docs)

        template = """
You are an AI assistant that explains India's National Education Policy (NEP 2020) and ABSS 2025.

Answer clearly and concisely using the context.

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


rag = SimplePolicyRAG()


@app.on_event("startup")
def startup_event():
    logger.info("Initializing system...")
    rag.load_documents()
    logger.info("System ready.")


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
        raise HTTPException(status_code=500, detail="Internal Server Error")
