"""Markdown-aware chunking for the k8s docs corpus.

Strategy (documented in README):
1. strip Hugo frontmatter (`---`) and shortcodes (`{{< ... >}}`)
2. split each page on headings (# / ## / ###) -> one chunk per section
3. sections longer than CHUNK_CHARS get recursive-split with overlap
Every chunk carries metadata: page title, section path, source url.
"""
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

CHUNK_CHARS = 900
OVERLAP = 135  # 15%

_splitter = RecursiveCharacterTextSplitter(
    chunk_size=CHUNK_CHARS,
    chunk_overlap=OVERLAP,
    separators=["\n\n", "\n", ". ", " ", ""],
)

def _strip_hugo(text: str) -> str:
    lines = text.splitlines()
    if lines and lines[0].strip() == "---":
        for i in range(1, len(lines)):
            if lines[i].strip() == "---":
                text = "\n".join(lines[i + 1:])
                break
    out = []
    for line in text.splitlines():
        s = line.strip()
        if s.startswith("{{<") or s.startswith("{{%"):
            continue
        out.append(line)
    return "\n".join(out)

def _sections(text: str):
    """Yield (heading_path, section_text) by splitting on #/##/### headings."""
    path: list[tuple[int, str]] = []
    current: list[str] = []
    for line in text.splitlines():
        if line.startswith("### "):
            if current:
                yield " > ".join(t for _, t in path), "\n".join(current).strip()
                current = []
            path = path[:2] + [(3, line[4:].strip())]
            continue
        if line.startswith("## "):
            if current:
                yield " > ".join(t for _, t in path), "\n".join(current).strip()
                current = []
            path = path[:1] + [(2, line[3:].strip())]
            continue
        if line.startswith("# "):
            if current:
                yield " > ".join(t for _, t in path), "\n".join(current).strip()
                current = []
            path = [(1, line[2:].strip())]
            continue
        current.append(line)
    if current:
        yield " > ".join(t for _, t in path), "\n".join(current).strip()

def chunk_page(title: str, url: str, raw: str) -> list[Document]:
    text = _strip_hugo(raw)
    docs = []
    for section_path, body in _sections(text):
        if not body:
            continue
        meta = {"title": title, "section": section_path, "source": url}
        pieces = _splitter.split_text(body)
        for i, piece in enumerate(pieces):
            docs.append(Document(page_content=piece, metadata={**meta, "part": i}))
    return docs
