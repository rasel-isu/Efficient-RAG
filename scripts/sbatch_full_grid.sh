#!/bin/bash
#SBATCH --job-name=erag-grid
#SBATCH --partition=nova
#SBATCH --gres=gpu:a100:4
#SBATCH --cpus-per-task=32
#SBATCH --mem=256G
#SBATCH --time=24:00:00
#SBATCH --output=logs/slurm-%j.out
#SBATCH --error=logs/slurm-%j.out
#
# Full (dataset x condition x rerank) grid, one worker per GPU.
#
# Why these resources: a single A100 runs at 100% utilisation on this workload,
# so throughput scales ~linearly with GPU count (30 cells at n=500 is ~33 h on
# one GPU, ~9-13 h on four). Each worker holds its own 32 GB index copy plus
# ~10 GB of models, which fits an 80 GB A100. The CPU and RAM requests matter as
# much as the GPUs: tokenisation is CPU-bound, and 2 cores (this project's
# interactive default) cannot feed four workers.
#
# Submit:   sbatch scripts/sbatch_full_grid.sh
# Resume:   sbatch scripts/sbatch_full_grid.sh      (finished cells are skipped)
# Watch:    tail -f logs/slurm-<jobid>.out ; tail -f logs/multigpu/gpu0.log

set -uo pipefail

PROJECT=/lustre/hdd/LAS/qli-lab/rasel/projects/Efficient-RAG
OUT_DIR=${OUT_DIR:-OUTPUT/full}
N=${N:-500}
TOP_K=${TOP_K:-30}
READER=${READER:-meta-llama/Llama-3.2-3B-Instruct}
SUMMARY_MODEL=${SUMMARY_MODEL:-google/flan-t5-small}
DATASETS=${DATASETS:-"hotpotqa 2wiki musique nq_open triviaqa"}
CONDITIONS=${CONDITIONS:-"baseline truncate filter_summ"}
RERANK=${RERANK:-"off on"}

eval "$(/lustre/hdd/LAS/qli-lab/rasel/apps/miniconda3/bin/conda shell.bash hook)"
source /lustre/hdd/LAS/qli-lab/rasel/apps/miniconda3/etc/profile.d/conda.sh
conda activate "$PROJECT/conda_env" || { echo "conda activate failed"; exit 1; }
cd "$PROJECT" || exit 1

mkdir -p logs logs/multigpu "$OUT_DIR"

echo "=== job ${SLURM_JOB_ID:-local} on $(hostname) at $(date) ==="
echo "GPUs visible : ${CUDA_VISIBLE_DEVICES:-unset}"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader
echo "datasets   : $DATASETS"
echo "conditions : $CONDITIONS   rerank: $RERANK"
echo "n=$N top_k=$TOP_K reader=$READER"
echo "out        : $OUT_DIR"

# --- preflight -------------------------------------------------------------
INDEX=DATASET/wiki_dpr/index_contriever/embeddings.fp16.npy
[ -s "$INDEX" ] || { echo "FATAL: missing $INDEX (run scripts/build_wiki_index.py)"; exit 1; }
for d in $DATASETS; do
  case $d in
    hotpotqa) f=DATASET/hotpot_qa/validation.parquet ;;
    2wiki)    f=DATASET/2WikiMultihopQA/validation.parquet ;;
    musique)  f=DATASET/musique/validation.parquet ;;
    nq_open)  f=DATASET/nq_open/validation.parquet ;;
    triviaqa) f=DATASET/trivia_qa/validation.parquet ;;
    *) echo "FATAL: unknown dataset $d"; exit 1 ;;
  esac
  [ -s "$f" ] || { echo "FATAL: missing $f"; exit 1; }
done

# Node-local scratch must hold the 32 GB index; fall back to the shared
# filesystem automatically if it cannot (the run is then slower, not wrong).
NEED_GB=$(( $(stat -c %s "$INDEX") / 1000000000 + 5 ))
FREE_GB=$(df -BG --output=avail /tmp 2>/dev/null | tail -1 | tr -dc '0-9')
echo "local scratch: ${FREE_GB:-?} GB free, need ~${NEED_GB} GB"

# Stage ONCE here rather than letting every worker copy it concurrently.
echo "=== staging index ($(date +%H:%M:%S)) ==="
python scripts/stage_index.py contriever || echo "WARN: staging failed, workers will read shared storage"

# --- run -------------------------------------------------------------------
echo "=== grid start ($(date +%H:%M:%S)) ==="
python scripts/run_multigpu.py \
  --datasets $DATASETS \
  --conditions $CONDITIONS \
  --rerank $RERANK \
  --n "$N" --top-k "$TOP_K" \
  --reader "$READER" \
  --summary-model "$SUMMARY_MODEL" \
  --out-dir "$OUT_DIR" \
  --log-dir logs/multigpu
RC=$?
echo "=== grid finished rc=$RC ($(date +%H:%M:%S)) ==="

# --- report ----------------------------------------------------------------
NCELLS=$(ls "$OUT_DIR"/*.json 2>/dev/null | wc -l)
echo "cells written: $NCELLS / 30"
if [ "$NCELLS" -gt 0 ]; then
  echo "=== console summary ==="
  python scripts/summarize_grid.py "$OUT_DIR/*.json" || true

  # Persistent reports. Written even on a partial grid so a preempted or
  # timed-out run still leaves something readable behind.
  GRID_REPORT=${GRID_REPORT:-RESULTS_full_grid.md}
  echo "=== grid analysis report ($(date +%H:%M:%S)) ==="
  python scripts/analyze_grid.py --glob "$OUT_DIR/*.json" --out "$GRID_REPORT" \
    && echo "report written: $PWD/$GRID_REPORT" \
    || echo "WARN: grid analysis failed (raw results in $OUT_DIR are unaffected)"

  if ls "$OUT_DIR"/*_baseline_rerank-off.json >/dev/null 2>&1; then
    echo "=== reproduction gate + baseline report ($(date +%H:%M:%S)) ==="
    python scripts/reproduce_compact.py \
      --results-glob "$OUT_DIR/*_baseline_rerank-off.json" || true
    REPORT=${REPORT:-RESULTS_reproduction.md}
    python scripts/analyze_reproduction.py \
      --glob "$OUT_DIR/*_baseline_rerank-off.json" --out "$REPORT" \
      && echo "report written: $PWD/$REPORT" || true
  fi
fi
echo "=== done $(date) ==="
exit $RC
