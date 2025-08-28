from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
from typing import List

from .models import AddDocumentRequest, SearchResult, ChatRequest, ChatResponse
from .database import ChromaDBManager
from .embeddings import EmbeddingGenerator
from .rag import RAGPipeline
from .config import OPENROUTER_API_KEY, CHROMADB_PATH, EMBEDDING_MODEL, MODEL_SLUG

app = FastAPI(title="RAG System API")

# Enable CORS for browser testing
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

db_manager = ChromaDBManager()
embedding_generator = EmbeddingGenerator(EMBEDDING_MODEL)
rag_pipeline = RAGPipeline()


@app.get("/")
def root():
    return {"name": "RAG System", "version": "1.0"}


@app.post("/add_document")
async def add_document(request: AddDocumentRequest):
    # TODO: Generate embedding and store in ChromaDB
    return {"success": True, "id": "doc_id"}


@app.get("/search", response_model=List[SearchResult])
async def search(
    query: str = Query(...),
    limit: int = Query(default=5)
):
    # TODO: Generate query embedding and search ChromaDB
    return []


@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    # TODO: Implement RAG
    result = rag_pipeline.generate_answer(
        request.question,
        request.max_results
    )
    
    return ChatResponse(**result)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)