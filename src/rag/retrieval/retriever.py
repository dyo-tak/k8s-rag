"""Retrieval module: MMR vector search over the k8s docs Chroma index."""
import sys
from functools import lru_cache
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from chromadb.utils.embedding_functions import ONNXMiniLM_L6_V2  # noqa: E402
from langchain_chroma import Chroma  # noqa: E402
from langchain_core.documents import Document  # noqa: E402

PERSIST = ROOT / "chroma_db"

@lru_cache(maxsize=1)
def get_vectorstore() -> Chroma:
    return Chroma(
        persist_directory=str(PERSIST),
        embedding_function=ONNXMiniLM_L6_V2(),
        collection_name="k8s_docs",
    )

def retrieve(question: str, k: int = 5, fetch_k: int = 25) -> list[Document]:
    """MMR retrieval: relevant AND diverse chunks."""
    return get_vectorstore().as_retriever(
        search_type="mmr",
        search_kwargs={"k": k, "fetch_k": fetch_k, "lambda_mult": 0.5},
    ).invoke(question)

def format_context(docs: list[Document]) -> str:
    return "\n\n---\n\n".join(
        f"[{d.metadata.get('title','')} :: {d.metadata.get('section','')}]\n{d.page_content}"
        for d in docs
    )
