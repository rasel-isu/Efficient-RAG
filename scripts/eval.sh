eval $(/lustre/hdd/LAS/qli-lab/rasel/apps/miniconda3/bin/conda shell.bash hook)
source /lustre/hdd/LAS/qli-lab/rasel/apps/miniconda3/etc/profile.d/conda.sh
python evaluation.py OUTPUT/rag-mini-wikipedia/gpt-3.5-turbo/t5_large_sumry.json > report.json