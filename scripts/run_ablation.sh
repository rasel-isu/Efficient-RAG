#!/bin/bash
# Full rag-mini-wikipedia ablation on the fixed pipeline.
# Sequential: embedded Weaviate binds a fixed port, so runs cannot overlap.
eval $(/lustre/hdd/LAS/qli-lab/rasel/apps/miniconda3/bin/conda shell.bash hook)
source /lustre/hdd/LAS/qli-lab/rasel/apps/miniconda3/etc/profile.d/conda.sh
conda activate /lustre/hdd/LAS/qli-lab/rasel/projects/Efficient-RAG/conda_env
cd /lustre/hdd/LAS/qli-lab/rasel/projects/Efficient-RAG
OUT=OUTPUT/rag-mini-wikipedia/Llama-3.2-3B-Instruct-fixed

run () {  # name, condition, model
  echo "=== $1 ($(date +%H:%M:%S)) ==="
  if [ -n "$3" ]; then
    python main.py --condition "$2" --summary-model "$3" --outfile "$OUT/$1.json" 2>&1 | grep -vE '^\{|ResourceWarning|^sys:1'
  else
    python main.py --condition "$2" --outfile "$OUT/$1.json" 2>&1 | grep -vE '^\{|ResourceWarning|^sys:1'
  fi
}

run baseline        baseline        ""
run keyword_only    keyword_only    ""
run summarizer_only summarizer_only google/flan-t5-small
run t5_small        filter_summ     google/flan-t5-small
run t5_base         filter_summ     google/flan-t5-base
run t5_large        filter_summ     google/flan-t5-large
echo "=== ALL DONE $(date +%H:%M:%S) ==="
