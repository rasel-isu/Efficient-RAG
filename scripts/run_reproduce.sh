#!/bin/bash
# Tier 1: reproduce CompAct's Raw-Document row before running anything else.
# Protocol: Contriever-MSMARCO / DPR Wikipedia 2018 / top-30 / LLaMA3-8B / no rerank.
eval $(/lustre/hdd/LAS/qli-lab/rasel/apps/miniconda3/bin/conda shell.bash hook)
source /lustre/hdd/LAS/qli-lab/rasel/apps/miniconda3/etc/profile.d/conda.sh
conda activate /lustre/hdd/LAS/qli-lab/rasel/projects/Efficient-RAG/conda_env
cd /lustre/hdd/LAS/qli-lab/rasel/projects/Efficient-RAG

READER=${READER:-meta-llama/Llama-3.2-3B-Instruct}
N=${N:-500}
OUT_DIR=${OUT_DIR:-OUTPUT/reproduce}
OUT=$OUT_DIR
mkdir -p "$OUT"

for DS in hotpotqa 2wiki musique nq_open triviaqa; do
  F="$OUT/${DS}_raw.json"
  [ -s "$F" ] && { echo "skip $DS"; continue; }
  echo "=== $DS raw-document ($(date +%H:%M:%S)) ==="
  python run_benchmark.py --dataset "$DS" --condition baseline \
    --retrieval corpus --encoder contriever --no-rerank \
    --top-k 30 --n "$N" --reader "$READER" --outfile "$F" \
    2>&1 | grep -vE '^\{|checkpoint shards|pad_token|torch_dtype|ResourceWarning'
done
python scripts/reproduce_compact.py --results-glob "$OUT/*_raw.json"
