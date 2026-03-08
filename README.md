# GitaVanni - Bhagavad Gita Intelligent RAG

GitaVanni is an AI-powered spiritual guide application that uses a Hybrid Retrieval-Augmented Generation (RAG) system to answer questions and provide insights based on the Bhagavad Gita. It combines deterministic exact verse retrieval with semantic search to provide highly accurate and spiritually context-aware answers.

## 🌟 Features

*   **Hybrid Search System:** Utilizes both FAISS (dense vector search) and BM25 (sparse keyword search) for optimal context retrieval.
*   **Deterministic Verse Retrieval:** If a user asks for a specific verse (e.g., "Chapter 2 Verse 47"), the system accurately fetches the exact Sanskrit verse directly from the source text before augmenting it with LLM insights.
*   **Beautiful UI (GitaVanni 3.0):** A responsive, spiritually-themed web interface with chat history, animations, and a seamless chat experience.
*   **Fast Inference:** Powered by Groq (`llama-3.3-70b-versatile`) for lightning-fast and intelligent responses.
*   **Dual-Source Knowledge Base:** Reads from both an English translation PDF (`gita_english.pdf`) and a structured Sanskrit text file (`bhagwatgita.txt`).

## 🛠️ Tech Stack

*   **Backend:** FastAPI, Python
*   **RAG Pipeline:** LangChain
*   **Vector Store:** FAISS
*   **Keyword Search:** BM25Retriever
*   **Embeddings:** HuggingFace (`all-MiniLM-L6-v2`)
*   **LLM Engine:** ChatGroq (`llama-3.3-70b-versatile`)
*   **Frontend:** Vanilla HTML, CSS, JavaScript (`index.html`)

## 📁 Project Structure

```
VANNI/
├── app.py                  # Main FastAPI application and RAG backend
├── index.html              # Frontend UI portal
├── .env                    # Environment variables (API keys)
├── gita_english.pdf        # Knowledge base source (English meaning)
├── bhagwatgita.txt         # Knowledge base source (Sanskrit verses)
├── faiss_index/            # Persisted local FAISS vector store
├── requirements.txt        # Python dependencies (create if not present)
└── README.md               # Project documentation
```

## 🚀 Getting Started

### Prerequisites

*   Python 3.8 or higher
*   A Groq API Key

### Installation

1.  **Clone the repository / navigate to the folder:**
    ```bash
    cd <path-to-VANNI-folder>
    ```

2.  **Create and activate a virtual environment (recommended):**
    ```bash
    python -m venv venv
    # Windows
    venv\Scripts\activate
    # macOS/Linux
    source venv/bin/activate
    ```

3.  **Install dependencies:**
    You will need the following key libraries installed:
    ```bash
    pip install fastapi uvicorn langchain langchain-community langchain-huggingface langchain-groq pydantic faiss-cpu rank_bm25 pypdf sentence-transformers
    ```

4.  **Set up Environment Variables:**
    Edit the `.env` file in the root directory and add your Groq API key:
    ```env
    GROQ_API_KEY=your_groq_api_key_here
    ```

### Running the Application

Start the FastAPI server using Uvicorn:

```bash
uvicorn app:app --reload
```

The application will start on `http://127.0.0.1:8000`.
Visit this URL in your web browser to open the GitaVanni 3.0 UI and start chatting!

## 🧠 How it Works

1.  **Initialization:** On the first run, the app loads `gita_english.pdf` and parses `bhagwatgita.txt` into chunks and document objects.
2.  **Indexing:** It then embeds these documents using HuggingFace sentence transformers, saves them locally to `faiss_index/`, and initializes a BM25 retriever. Subsequent runs load the index instantly from disk.
3.  **Querying:** User queries are analyzed to see if they reference a specific verse (e.g., "3.14" or "Chapter 3 verse 14").
    *   *If exact:* The system strictly outputs the Sanskrit Shloka from the TXT logic, its translation, and a contextual takeaway.
    *   *If general:* The hybrid retriever grabs top-K relevant passages using FAISS+BM25 and frames a spiritual answer without hallucinating, keeping Sanskrit usage minimal.
