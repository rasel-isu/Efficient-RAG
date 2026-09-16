#!/bin/bash
# Wait for corpus sharding, smoke-test the encoder, then commit the full pass.
# Fails fast so a broken encoder costs seconds, not 4 GPU-hours.
set -e
eval $(/lustre/hdd/LAS/qli-lab/rasel/apps/miniconda3/bin/conda shell.bash hook)
source /lustre/hdd/LAS/qli-lab/rasel/apps/miniconda3/etc/profile.d/conda.sh
conda activate /lustre/hdd/LAS/qli-lab/rasel/projects/Efficient-RAG/conda_env
cd /lustre/hdd/LAS/qli-lab/rasel/projects/Efficient-RAG

echo "=== waiting for corpus shards ($(date +%H:%M:%S)) ==="
while ps -p ${1:-0} > /dev/null 2>&1; do sleep 30; done
ls -la DATASET/wiki_dpr/
python - <<'PY'
import glob, pandas as pd
fs = sorted(glob.glob("DATASET/wiki_dpr/corpus-*.parquet"))
n = sum(pd.read_parquet(f, columns=["id"]).shape[0] for f in fs)
print(f"shards={len(fs)} total_passages={n:,}")
assert n > 20_000_000, f"expected ~21M passages, got {n:,}"
PY

echo "=== encoder smoke test ($(date +%H:%M:%S)) ==="
python scripts/build_wiki_index.py --encoder contriever --limit 20000 \
  --corpus-glob 'DATASET/wiki_dpr/corpus-00.parquet'
echo "=== smoke OK - starting FULL contriever pass ($(date +%H:%M:%S)) ==="
python scripts/build_wiki_index.py --encoder contriever
echo "=== CONTRIEVER INDEX DONE $(date +%H:%M:%S) ==="
