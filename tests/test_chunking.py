"""Chunking unit tests — no LLM, no index, fast."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from rag.ingest.chunking import chunk_page

PAGE = """---
title: Pods
weight: 10
---

# Pods

Pods are the smallest deployable units.

## Container limits

You can set CPU and memory limits on containers.

{{< note >}}
shortcodes are stripped
{{< /note >}}

### Long section

{LONG}
"""

def test_chunks_have_metadata():
    raw = PAGE.replace("{LONG}", "word " * 50)
    docs = chunk_page("Pods", "https://kubernetes.io/docs/concepts/workloads/pods/", raw)
    assert docs, "expected chunks"
    for d in docs:
        assert d.metadata["title"] == "Pods"
        assert d.metadata["source"].startswith("https://kubernetes.io/docs/")
        assert d.page_content.strip()

def test_shortcodes_stripped():
    raw = PAGE.replace("{LONG}", "word " * 50)
    docs = chunk_page("Pods", "u", raw)
    joined = "\n".join(d.page_content for d in docs)
    assert "{{<" not in joined
    assert "frontmatter" not in joined

def test_long_sections_are_split():
    raw = PAGE.replace("{LONG}", "sentence about scheduling. " * 300)
    docs = chunk_page("Pods", "u", raw)
    long_section = [d for d in docs if "Long section" in d.metadata["section"]]
    assert len(long_section) >= 2, "oversized section should be recursive-split"
