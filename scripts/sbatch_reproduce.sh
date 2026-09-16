#!/bin/bash
#SBATCH --job-name=erag-repro
#SBATCH --partition=nova
#SBATCH --gres=gpu:a100:2
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=16:00:00
#SBATCH --output=logs/slurm-%j.out
#SBATCH --error=logs/slurm-%j.out
#
# Tier-1 gate: runs the COMPRESSED conditions plus the uncompressed reference,
# on all five datasets, before committing to the full grid.
#
# The uncompressed condition is not the contribution - it is the DENOMINATOR.
# "88% token reduction" and "95% accuracy retention" are both ratios against it,
# so it has to be measured in the same setting or neither number exists. It also
# doubles as the check against CompAct Table 2's "Raw Document" row.
#
# Protocol: Contriever-MSMARCO over DPR Wikipedia 2018, top-30, NO reranking
# (none of the baselines rerank).
#
# Absolute EM will NOT match CompAct: they use LLaMA3-8B, this uses
# Llama-3.2-3B. The gate therefore checks that the ORDERING of dataset
# difficulty is preserved (Spearman rho >= 0.8), which a correct pipeline
# keeps and a broken one does not.
#
# Submit: sbatch scripts/sbatch_reproduce.sh

set -uo pipefail

PROJECT=/lustre/hdd/LAS/qli-lab/rasel/projects/Efficient-RAG
OUT_DIR=${OUT_DIR:-OUTPUT/reproduce}
N=${N:-500}
READER=${READER:-meta-llama/Llama-3.2-3B-Instruct}
SUMMARY_MODEL=${SUMMARY_MODEL:-google/flan-t5-small}
# filter_summ runs first so `truncate` can be matched to the budget it produced
CONDITIONS=${CONDITIONS:-"baseline truncate filter_summ"}
RERANK=${RERANK:-"off"}

eval "$(/lustre/hdd/LAS/qli-lab/rasel/apps/miniconda3/bin/conda shell.bash hook)"
source /lustre/hdd/LAS/qli-lab/rasel/apps/miniconda3/etc/profile.d/conda.sh
conda activate "$PROJECT/conda_env" || { echo "conda activate failed"; exit 1; }
cd "$PROJECT" || exit 1
mkdir -p logs logs/multigpu "$OUT_DIR"

echo "=== job ${SLURM_JOB_ID:-local} on $(hostname) at $(date) ==="
nvidia-smi --query-gpu=index,name --format=csv,noheader

INDEX=DATASET/wiki_dpr/index_contriever/embeddings.fp16.npy
[ -s "$INDEX" ] || { echo "FATAL: missing $INDEX"; exit 1; }

echo "=== staging index ($(date +%H:%M:%S)) ==="
python scripts/stage_index.py contriever || echo "WARN: staging failed, reading shared storage"

echo "=== reproduction start ($(date +%H:%M:%S)) ==="
python scripts/run_multigpu.py \
  --datasets hotpotqa 2wiki musique nq_open triviaqa \
  --conditions $CONDITIONS \
  --rerank $RERANK \
  --n "$N" --top-k 30 \
  --reader "$READER" \
  --summary-model "$SUMMARY_MODEL" \
  --out-dir "$OUT_DIR" \
  --log-dir logs/multigpu
RC=$?

echo "=== gate verdict ($(date +%H:%M:%S)) ==="
python scripts/reproduce_compact.py --results-glob "$OUT_DIR/*_baseline_rerank-off.json"
GATE=$?

# Always write the persistent analysis report - the gate block above only
# survives in this slurm log, which is easy to lose track of.
REPORT=${REPORT:-RESULTS_reproduction.md}
echo "=== analysis report ($(date +%H:%M:%S)) ==="
python scripts/analyze_reproduction.py \
  --glob "$OUT_DIR/*_baseline_rerank-off.json" \
  --out "$REPORT" && echo "report written: $PWD/$REPORT" \
  || echo "WARN: baseline analysis failed"

GRID_REPORT=${GRID_REPORT:-RESULTS_reproduction_grid.md}
python scripts/analyze_grid.py --glob "$OUT_DIR/*.json" --out "$GRID_REPORT" \
  && echo "report written: $PWD/$GRID_REPORT (token reduction + retention)" \
  || echo "WARN: grid analysis failed (raw results in $OUT_DIR are unaffected)"
echo "cells: $(ls "$OUT_DIR"/*.json 2>/dev/null | wc -l)/15   run_rc=$RC  gate_rc=$GATE"
echo "gate_rc 0 = ordering preserved, 1 = diverged (inspect before the full grid)"
echo "=== done $(date) ==="
exit $RC
