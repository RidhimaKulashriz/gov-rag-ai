# app.py
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from src.rag_pipeline import RAGPipeline

# Initialize FastAPI
app = FastAPI(
    title="Government Document Intelligence API",
    description="RAG-based querying of government documents",
    version="1.0"
)

# Enable CORS for frontend connections (optional)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # adjust to your frontend domain in production
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize RAG pipeline
rag = RAGPipeline()

@app.get("/query")
def query(q: str, top_k: int = 10):
    """
    Query endpoint for RAG pipeline.
    Example: /query?q=What+is+the+trauma+protocol?
    
    Parameters:
    - q: str → question to ask
    - top_k: int → number of top documents to retrieve (default 10)
    
    Returns:
    JSON response containing:
    - answer: generated answer
    """
    if not q:
        raise HTTPException(status_code=400, detail="Query parameter 'q' is required")
    
    answer_text = rag.answer(question=q, top_k=top_k)
    return {"question": q, "answer": answer_text, "top_k": top_k}

# Health check endpoint
@app.get("/health")
def health():
    return {"status": "ok"}