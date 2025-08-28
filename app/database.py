from __future__ import annotations
import os
from typing import List, Dict, Any, Optional
import chromadb
from chromadb import Collection
from chromadb.config import Settings

class VectorDB:
    def __init__(self, collection_name: str = "documents", persist_dir_env: str = "CHROMADB_PATH") -> None:
        persist_dir = os.getenv(persist_dir_env, "./chroma_db")
        self._client = chromadb.PersistentClient(path=persist_dir, settings=Settings(allow_reset=False))
        self._collection: Collection = self._client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
        )

    @property
    def collection(self) -> Collection:
        return self._collection

    def add_texts(
        self,
        texts: List[str],
        ids: Optional[List[str]] = None,
        metadatas: Optional[List[Dict[str, Any]]] = None,
        embeddings: Optional[List[List[float]]] = None,
    ) -> None:
        if not texts:
            raise ValueError("Lista de textos vazia.")
        if ids and len(ids) != len(texts):
            raise ValueError("ids e texts devem ter o mesmo tamanho.")
        if metadatas and len(metadatas) != len(texts):
            raise ValueError("metadatas e texts devem ter o mesmo tamanho.")
        if embeddings and len(embeddings) != len(texts):
            raise ValueError("embeddings e texts devem ter o mesmo tamanho.")
        if ids is None:
            ids = [f"doc-{i}" for i in range(len(texts))]
        self._collection.add(documents=texts, ids=ids, metadatas=metadatas, embeddings=embeddings)

    def search(self, query_embedding: List[float], k: int = 5) -> Dict[str, Any]:
        if k <= 0 or k > 25:
            k = 5
        return self._collection.query(query_embeddings=[query_embedding], n_results=k)
    # Alias para compatibilidade com código que importa ChromaDBManager
ChromaDBManager = VectorDB

