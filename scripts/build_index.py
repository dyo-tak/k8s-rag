"""Build the Chroma index from data/k8s_docs (local MiniLM embeddings, no API key).

Run: uv run python scripts/build_index.py
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from chromadb.utils.embedding_functions import ONNXMiniLM_L6_V2  # noqa: E402
from langchain_chroma import Chroma  # noqa: E402

from rag.ingest.chunking import chunk_page  # noqa: E402

PERSIST = ROOT / "chroma_db"

def load_pages():
    pages = []
    for md in sorted((ROOT / "data" / "k8s_docs").rglob("*.md")):
        rel = md.relative_to(ROOT / "data" / "k8s_docs").as_posix()
        url = "https://kubernetes.io/docs/" + rel.removesuffix(".md").removesuffix("/_index")
        title = next(
            (l.lstrip("# ").strip() for l in md.read_text().splitlines() if l.startswith("# ")),
            rel,
        )
        pages.append((title, url, md.read_text()))
    return pages

def main() -> int:
    pages = load_pages()
    if not pages:
        print("no docs found — run scripts/fetch_docs.py first", file=sys.stderr)
        return 1
    all_chunks = []
    for title, url, raw in pages:
        all_chunks.extend(chunk_page(title, url, raw))
    print(f"{len(pages)} pages -> {len(all_chunks)} chunks")

    vs = Chroma.from_documents(
        all_chunks,
        embedding=ONNXMiniLM_L6_V2(),
        persist_directory=str(PERSIST),
        collection_name="k8s_docs",
    )
    print(f"indexed {vs._collection.count()} chunks -> {PERSIST}")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
