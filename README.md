
# Technical Test: Simple RAG System (Entrega – Flávio Antonio Oliveira da Silva)

Este repositório contém minha implementação do desafio técnico de **Retrieval-Augmented Generation (RAG)** solicitado pela Action Labs.

A API foi construída em **Python + FastAPI**, utilizando **ChromaDB** como vetor DB e embeddings com **Sentence Transformers**.  
Também inclui **modo offline** (sem internet/sem chave de API) para facilitar a avaliação.

---

## ⚙️ Tecnologias
- [FastAPI](https://fastapi.tiangolo.com/)
- [Uvicorn](https://www.uvicorn.org/)
- [ChromaDB](https://docs.trychroma.com/)
- [Sentence Transformers](https://www.sbert.net/)
- [httpx](https://www.python-httpx.org/)

---

## ▶️ Como rodar

### 1. Clonar e instalar dependências
```bash
git clone https://github.com/flaviocaos/llm-rag-test.git
cd llm-rag-test
python -m venv .venv
source .venv/Scripts/activate   # (Windows Git Bash)
pip install -r requirements.txt


2. Rodar em modo offline (sem chave de API)
export FAKE_EMBEDDINGS=1
export FAKE_LLM=1
export CHROMADB_PATH=./chroma_db
uvicorn app.main:app --reload


Acesse: http://localhost:8000/docs

3. Rodar em modo online (com OpenRouter)

Crie um .env:

OPENROUTER_API_KEY=coloque_sua_chave_aqui
MODEL_SLUG=openai/gpt-3.5-turbo
CHROMADB_PATH=./chroma_db
FAKE_LLM=0
FAKE_EMBEDDINGS=0


E rode:

uvicorn app.main:app --reload

Endpoints
POST /add_document
{
  "text": "A Action Labs é especializada em IA aplicada.",
  "metadata": {"source": "about_action_labs"}
}

GET /search
query=Em que a Action Labs é boa?&k=3

POST /chat
{
  "question": "Qual é a especialidade da empresa?",
  "k": 3
}

Extras implementados

Modo offline com FAKE_EMBEDDINGS e FAKE_LLM

Compatibilidade com imports originais (ChromaDBManager, EmbeddingGenerator)

Testes automatizados (pytest)

Dockerfile para container

CI no GitHub Actions (lint + testes + build)

Autor: Flávio Antonio Oliveira da Silva
Entrega: Processo Seletivo Action Labs
