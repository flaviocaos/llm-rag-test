from __future__ import annotations
import os
from typing import List, Dict, Any, Tuple
import httpx

from .database import VectorDB
from .embeddings import Embedder

_OPENROUTER_KEY_ENV = "OPENROUTER_API_KEY"
_MODEL_SLUG_ENV = "MODEL_SLUG"
_DEFAULT_MODEL_SLUG = "openai/gpt-3.5-turbo"

SYSTEM_PROMPT = """You are a helpful assistant. Answer the user based ONLY on the provided context.
If the answer is not in the context, say you don't know. Respond objectively and cite the sources.
"""

USER_TEMPLATE = """Question: {question}

Context:
{context}

Instructions:
- Use only the context to answer.
- Cite sources as [source_i] ao final.
"""

class RAGPipeline:
    def __init__(self) -> None:
        self.db = VectorDB()
        self.embedder = Embedder()
        self.model_slug = os.getenv(_MODEL_SLUG_ENV, _DEFAULT_MODEL_SLUG)

        # OFFLINE
        self.fake_llm = os.getenv("FAKE_LLM", "0") == "1"

        self.api_key = os.getenv(_OPENROUTER_KEY_ENV)
        if not self.fake_llm and not self.api_key:
            raise RuntimeError("OPENROUTER_API_KEY não configurada (ou use FAKE_LLM=1).")

    def add_document(self, text: str, metadata: Dict[str, Any] | None = None) -> Dict[str, Any]:
        text = (text or "").strip()
        if not text:
            raise ValueError("Documento vazio.")
        emb = self.embedder.embed(text)
        self.db.add_texts([text], metadatas=[metadata or {}], embeddings=[emb])
        return {"ok": True}

    def search(self, query: str, k: int = 5) -> Dict[str, Any]:
        q_emb = self.embedder.embed(query)
        results = self.db.search(q_emb, k=k)
        out = []
        docs = results.get("documents", [[]])[0]
        ids = results.get("ids", [[]])[0]
        metas = results.get("metadatas", [[]])[0]
        dists = results.get("distances", [[]])[0]
        for i, doc in enumerate(docs):
            out.append({
                "id": ids[i] if i < len(ids) else None,
                "text": doc,
                "metadata": metas[i] if i < len(metas) else {},
                "score": dists[i] if i < len(dists) else None
            })
        return {"matches": out}

    def chat(self, question: str, k: int = 5) -> Dict[str, Any]:
        retrieved = self.search(question, k=k)["matches"]
        context, sources = self._build_context(retrieved)

        if self.fake_llm:
            if not context.strip():
                answer = "Não sei com base no contexto disponível."
            else:
                answer = f"Com base no contexto, a resposta é: {question} — veja {', '.join(s['label'] for s in sources)}."
            return {"answer": answer, "sources": sources}

        prompt = USER_TEMPLATE.format(question=question, context=context)
        answer = self._call_openrouter(SYSTEM_PROMPT, prompt)
        return {"answer": answer, "sources": sources}

    @staticmethod
    def _build_context(matches: List[Dict[str, Any]]) -> Tuple[str, List[Dict[str, Any]]]:
        MAX_CHARS = 3500
        snippets, sources, total = [], [], 0
        for i, m in enumerate(matches):
            txt: str = m.get("text", "")
            meta = m.get("metadata") or {}
            sid = meta.get("source") or m.get("id") or f"chunk_{i}"
            snippet = txt.strip()
            if not snippet:
                continue
            if total + len(snippet) > MAX_CHARS:
                snippet = snippet[: max(0, MAX_CHARS - total)]
            total += len(snippet)
            snippets.append(f"[source_{i}] {snippet}")
            sources.append({"id": sid, "label": f"source_{i}", "metadata": meta})
            if total >= MAX_CHARS:
                break
        return "\n\n".join(snippets), sources

    def _call_openrouter(self, system: str, user: str) -> str:
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "HTTP-Referer": "http://localhost",
            "X-Title": "llm-rag-test",
            "Content-Type": "application/json",
        }
        payload = {
            "model": self.model_slug,
            "messages": [{"role": "system", "content": system},
                         {"role": "user", "content": user}],
            "temperature": 0.1,
            "max_tokens": 500
        }
        url = "https://openrouter.ai/api/v1/chat/completions"
        timeout = httpx.Timeout(20.0, connect=10.0)
        with httpx.Client(timeout=timeout) as client:
            resp = client.post(url, headers=headers, json=payload)
            resp.raise_for_status()
            data = resp.json()
            content = (data.get("choices", [{}])[0].get("message", {}).get("content", ""))
            return content.strip()
