#!/bin/bash
# The core experiment: does compression benefit depend on retrieval precision?
#
# Sweeps the number of distractors reaching the retriever (0 = oracle evidence,
# 8 = the noisy regime prior compression work evaluates in) crossed with three
# context conditions. Prediction under test: the accuracy gap between baseline
# and compressed narrows (or inverts) as retrieval gets cleaner.
eval $(/lustre/hdd/LAS/qli-lab/rasel/apps/miniconda3/bin/conda shell.bash hook)
source /lustre/hdd/LAS/qli-lab/rasel/apps/miniconda3/etc/profile.d/conda.sh
conda activate /lustre/hdd/LAS/qli-lab/rasel/projects/Efficient-RAG/conda_env
cd /lustre/hdd/LAS/qli-lab/rasel/projects/Efficient-RAG

DS=${1:-hotpotqa}
N=${2:-500}
TRUNC=${3:-200}      # calibrate against the measured T5 context length
OUT=OUTPUT/$DS/Llama-3.2-3B-Instruct
mkdir -p "$OUT"

for D in 0 2 4 8; do
  for C in baseline filter_summ truncate; do
    F="$OUT/${C}_d${D}.json"
    [ -s "$F" ] && { echo "skip $F"; continue; }
    echo "=== $DS $C d=$D ($(date +%H:%M:%S)) ==="
    ARGS="--dataset $DS --condition $C --n $N --n-distractors $D --outfile $F"
    [ "$C" = "filter_summ" ] && ARGS="$ARGS --summary-model google/flan-t5-small"
    [ "$C" = "truncate" ]    && ARGS="$ARGS --truncate-to $TRUNC"
    python run_benchmark.py $ARGS 2>&1 | grep -vE '^\{|ResourceWarning|^sys:1|pad_token|checkpoint shards'
  done
done
echo "=== SWEEP DONE $(date +%H:%M:%S) ==="
