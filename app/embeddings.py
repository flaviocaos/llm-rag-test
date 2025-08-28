from __future__ import annotations
import os
import math
from typing import List

_DEFAULT_MODEL_ENV = "EMBEDDING_MODEL"
_DEFAULT_MODEL = "all-MiniLM-L6-v2"

class Embedder:
    _model = None

    def __init__(self, model_name: str | None = None) -> None:
        self.model_name = model_name or os.getenv(_DEFAULT_MODEL_ENV, _DEFAULT_MODEL)
        self.fake = os.getenv("FAKE_EMBEDDINGS", "0") == "1"

    # -------- FAKE MODE (sem downloads) --------
    @staticmethod
    def _tok(text: str) -> List[str]:
        return [t for t in (text or "").lower().split() if t.strip()]

    @staticmethod
    def _hash_token(t: str, dim: int) -> int:
        return (hash(t) % dim + dim) % dim

    def _embed_fake(self, text: str, dim: int = 384) -> List[float]:
        toks = self._tok(text)
        if not toks:
            raise ValueError("Texto vazio para embedding.")
        vec = [0.0] * dim
        for t in toks:
            vec[self._hash_token(t, dim)] += 1.0
        norm = math.sqrt(sum(v * v for v in vec)) or 1.0
        return [v / norm for v in vec]

    def _embed_batch_fake(self, texts: List[str], dim: int = 384) -> List[List[float]]:
        return [self._embed_fake(t, dim=dim) for t in texts]

    # -------- REAL MODE (SentenceTransformer) --------
    def _ensure_model(self):
        if self.__class__._model is None:
            from sentence_transformers import SentenceTransformer  # import tardio
            self.__class__._model = SentenceTransformer(self.model_name)
        return self.__class__._model

    # -------- API --------
    def embed(self, text: str) -> List[float]:
        text = (text or "").strip()
        if not text:
            raise ValueError("Texto vazio para embedding.")
        if self.fake:
            return self._embed_fake(text)
        model = self._ensure_model()
        return model.encode(text, normalize_embeddings=True).tolist()

    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        if not texts:
            raise ValueError("Lista de textos vazia para embedding.")
        if self.fake:
            return self._embed_batch_fake([t.strip() for t in texts])
        model = self._ensure_model()
        vecs = model.encode([t.strip() for t in texts], normalize_embeddings=True)
        return [v.tolist() for v in vecs]
    # Alias para compatibilidade com código que importa EmbeddingGenerator
EmbeddingGenerator = Embedder

