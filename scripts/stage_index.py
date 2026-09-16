"""Pre-stage the index onto node-local scratch before workers start.

Each worker would otherwise stage it independently on a fresh compute node -
four simultaneous 32 GB copies from Lustre. Doing it once up front costs ~2 min
and every worker then finds the local copy.
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from rag.global_retrieval import _stage_locally

encoder = sys.argv[1] if len(sys.argv) > 1 else "contriever"
src = f"DATASET/wiki_dpr/index_{encoder}/embeddings.fp16.npy"
if not os.path.exists(src):
    raise SystemExit(f"missing {src}")
dst = _stage_locally(src)
print(f"[stage] reads will come from: {dst}")
print("[stage] LOCAL" if dst != src else "[stage] FALLBACK to shared filesystem")
