"""Retrieval-only eval: does the right doc land in top-k? (no LLM needed)

Run: uv run python scripts/eval_retrieval.py
Writes eval/results.json — this is the number that goes in the README and
gets gated in CI later.
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from rag.retrieval.retriever import retrieve  # noqa: E402

# (question, expected substring in source path of at least one top-k chunk)
CASES = [
    ("why is my pod crashing with OOMKilled", "pods"),
    ("how do I set cpu and memory limits on a container", "assign-cpu-memory"),
    ("what is a taint and how do I make pods tolerate it", "taint-and-toleration"),
    ("how does a Deployment roll out updates", "deployment"),
    ("difference between ConfigMap and Secret", "configmap"),
    ("how do I expose my pods to the internet", "service"),
    ("what is a headless service", "service"),
    ("how does StatefulSet differ from Deployment", "statefulset"),
    ("what is a PersistentVolumeClaim", "persistent-volumes"),
    ("how does kube-scheduler pick a node", "kube-scheduler"),
    ("what is a daemonset used for", "daemonset"),
    ("how do CronJobs work", "cron-jobs"),
    ("what is an Ingress", "ingress"),
    ("how do I give a pod access to the cluster API", "service-accounts"),
    ("what is a ResourceQuota", "resource-quotas"),
    ("how do liveness and readiness probes work", "pod-lifecycle"),
    ("what is a NetworkPolicy", "network-policies"),
    ("how do I run a pod on specific nodes", "assign-pod-node"),
    ("what is a StorageClass", "storage-classes"),
    ("how do pod priority and preemption work", "pod-priority-preemption"),
]

def main() -> int:
    hits = 0
    results = []
    for q, expect in CASES:
        docs = retrieve(q, k=5)
        got = [d.metadata.get("source", "") for d in docs]
        hit = any(expect in s for s in got)
        hits += hit
        results.append({"question": q, "expected": expect, "hit": hit, "sources": got})
        print(f"{'PASS' if hit else 'MISS'}  {q}")
    score = hits / len(CASES)
    print(f"\nhit@5 = {hits}/{len(CASES)} = {score:.0%}")
    (ROOT / "eval").mkdir(exist_ok=True)
    (ROOT / "eval" / "results.json").write_text(json.dumps(
        {"hit_at_5": score, "cases": results}, indent=2))
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
