#!/bin/bash
# Wait for the Contriever index, sanity-check it, then run the Tier-1 reproduction.
set -e
eval $(/lustre/hdd/LAS/qli-lab/rasel/apps/miniconda3/bin/conda shell.bash hook)
source /lustre/hdd/LAS/qli-lab/rasel/apps/miniconda3/etc/profile.d/conda.sh
conda activate /lustre/hdd/LAS/qli-lab/rasel/projects/Efficient-RAG/conda_env
cd /lustre/hdd/LAS/qli-lab/rasel/projects/Efficient-RAG

echo "=== waiting for contriever index ($(date +%H:%M:%S)) ==="
while pgrep -f "build_wiki_index.py --encoder contriever$" > /dev/null; do sleep 60; done

python - <<'PY'
import numpy as np, pandas as pd
e = np.load("DATASET/wiki_dpr/index_contriever/embeddings.fp16.npy", mmap_mode="r")
p = pd.read_parquet("DATASET/wiki_dpr/index_contriever/passages.parquet", columns=["id"])
print(f"index shape={e.shape} passages={len(p):,}")
assert e.shape[0] == len(p), "index/passage length mismatch"
# the tail must not be all zeros - that would mean the run died mid-write
tail = np.asarray(e[-1000:], dtype=np.float32)
assert np.abs(tail).sum() > 0, "tail of index is all zeros - build incomplete"
print("index sanity OK")
PY

echo "=== Tier-1 reproduction, reader=Llama-3.2-3B ($(date +%H:%M:%S)) ==="
bash scripts/run_reproduce.sh
echo "=== REPRODUCTION DONE $(date +%H:%M:%S) ==="
