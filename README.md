# rag-bot

A local Retrieval-Augmented Generation (RAG) system built with FastAPI, vector embeddings, and a custom frontend interface.

## Tech Stack
- **Backend:** FastAPI, Python
- **Retrieval Engine:** Custom RAG pipeline (`backend/rag_engine.py`)
- **Vector Store:** ChromaDB (`backend/chroma_db/`)
- **Frontend:** HTML / Static UI (`frontend/`)

## Project Structure

├── backend/
│   ├── app.py           # FastAPI entrypoint and API routes
│   ├── rag_engine.py    # Document chunking, embedding, and retrieval logic
│   ├── chroma_db/       # Persistent vector database storage
│   └── requirements.txt # Python dependencies
└── frontend/
    └── index.html       # User interface files