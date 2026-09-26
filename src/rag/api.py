"""FastAPI service: POST /query, GET /health.

Run: uv run uvicorn src.rag.api:app --port 8000
"""
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from dotenv import load_dotenv
from fastapi import FastAPI
from pydantic import BaseModel, Field

from rag.chain import build_chain
from rag.retrieval.retriever import retrieve

load_dotenv(ROOT / ".env")

app = FastAPI(title="k8s-rag", version="0.1.0")

class Query(BaseModel):
    question: str = Field(min_length=3, max_length=500)
    k: int = Field(default=5, ge=1, le=10)

class Answer(BaseModel):
    question: str
    answer: str
    sources: list[str]

_chain = None

def get_chain():
    global _chain
    if _chain is None:
        from langchain_openai import ChatOpenAI
        llm = ChatOpenAI(  # OpenRouter-compatible
            model=os.getenv("OPENROUTER_MODEL", "openrouter/free"),
            base_url="https://openrouter.ai/api/v1",
            api_key=os.environ["OPENROUTER_API_KEY"],
            temperature=0,
        )
        _chain = build_chain(llm)
    return _chain

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/query", response_model=Answer)
def query(q: Query):
    docs = retrieve(q.question, k=q.k)
    answer = get_chain().invoke({"question": q.question})
    return Answer(
        question=q.question,
        answer=answer,
        sources=[d.metadata.get("source", "") for d in docs],
    )
