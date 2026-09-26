"""Download the Kubernetes docs corpus listed in manifest/docs_manifest.txt.

Run: uv run python scripts/fetch_docs.py
Docs are pinned by URL; failures are reported, not silently skipped.
"""
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "manifest" / "docs_manifest.txt"
OUT = ROOT / "data" / "k8s_docs"
BASE = "https://raw.githubusercontent.com/kubernetes/website/main/content/en/docs/"

def main() -> int:
    paths = [l.strip() for l in MANIFEST.read_text().splitlines() if l.strip() and not l.startswith("#")]
    ok = fail = 0
    for rel in paths:
        dest = OUT / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        url = BASE + rel
        try:
            with urllib.request.urlopen(url, timeout=30) as r:
                dest.write_bytes(r.read())
            ok += 1
            print(f"ok   {rel}")
        except Exception as e:  # noqa: BLE001
            fail += 1
            print(f"FAIL {rel}: {e}", file=sys.stderr)
    print(f"\n{ok} fetched, {fail} failed -> {OUT}")
    return 1 if fail else 0

if __name__ == "__main__":
    raise SystemExit(main())
