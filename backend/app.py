import os
import shutil
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from dotenv import load_dotenv

from rag_engine import RAGEngine

load_dotenv()

app = FastAPI(title="RAG Agent API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

rag = RAGEngine()
UPLOAD_DIR = "./temp_uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

# Build absolute path to frontend directory
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
FRONTEND_DIR = os.path.abspath(os.path.join(BASE_DIR, "..", "frontend"))
INDEX_PATH = os.path.join(FRONTEND_DIR, "index.html")

# Mount frontend directory for static assets (CSS, JS) if present
if os.path.exists(FRONTEND_DIR):
    app.mount("/static", StaticFiles(directory=FRONTEND_DIR), name="static")

class QueryRequest(BaseModel):
    question: str
    top_k: int = 4

@app.post("/api/upload")
async def upload_document(file: UploadFile = File(...)):
    allowed_extensions = (".pdf", ".txt", ".md")
    if not file.filename.lower().endswith(allowed_extensions):
        raise HTTPException(status_code=400, detail="Only PDF, TXT, and MD files are supported.")
    
    file_path = os.path.join(UPLOAD_DIR, file.filename)
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
        
    try:
        chunks_created = rag.process_and_ingest(file_path, file.filename)
        os.remove(file_path)
        return {"filename": file.filename, "chunks": chunks_created, "status": "Success"}
    except Exception as e:
        if os.path.exists(file_path):
            os.remove(file_path)
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/query")
async def query_knowledge_base(request: QueryRequest):
    if not request.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty.")
    return rag.query(request.question, request.top_k)

@app.get("/api/stats")
async def get_stats():
    return rag.get_stats()

@app.delete("/api/reset")
async def reset_knowledge_base():
    """Wipes vector embeddings, chunks, and database files."""
    try:
        rag.clear_database()
        return {"status": "Success", "message": "Knowledge base and store cleared."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Serve static frontend UI via FileResponse
@app.get("/")
async def serve_ui():
    if not os.path.exists(INDEX_PATH):
        raise HTTPException(status_code=404, detail=f"index.html not found at: {INDEX_PATH}")
    return FileResponse(INDEX_PATH)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)