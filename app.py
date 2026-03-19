from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from src.rag_pipeline import RAGPipeline

app = FastAPI(
    title="Government Document Intelligence API",
    description="RAG-based querying of government documents",
    version="1.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

rag = RAGPipeline()

@app.get("/query")
def query(q: str, top_k: int = 10):
    if not q:
        raise HTTPException(status_code=400, detail="Query parameter 'q' is required")
    answer_text = rag.answer(question=q, top_k=top_k)
    return {"question": q, "answer": answer_text, "top_k": top_k}

@app.get("/health")
def health():
    return {"status": "ok"}
