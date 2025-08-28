__version__ = "1.0.0"
__author__ = "Your Name"
__description__ = "Vector Database API with RAG capabilities"

# Ajustes de alias p/ manter compatibilidade com o restante do código
from .database import VectorDB as ChromaDBManager
from .embeddings import Embedder as EmbeddingGenerator
from .rag import RAGPipeline

__all__ = [
    "ChromaDBManager",
    "EmbeddingGenerator",
    "RAGPipeline",
]
