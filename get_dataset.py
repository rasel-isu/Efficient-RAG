
import pandas as pd
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
import textwrap
import os
from datasets import load_dataset

# dataset_name = "rag-datasets/rag-mini-bioasq"
# # dataset_name = "rag-datasets/rag-mini-wikipedia"
# dataset = load_dataset(dataset_name, "question-answer-passages")
# # dataset = load_dataset(dataset_name, "question-answer")
# # for split in dataset:
# #     df = dataset[split].to_pandas()
# #     output_file = f"DATASET/{dataset_name.replace('rag-datasets/', '')}/{split}.csv"
# #     df.to_csv(output_file, index=False)
# #     print(f"Saved {output_file} with shape {df.shape}")

# dataset = load_dataset("rag-datasets/rag-mini-bioasq", "text-corpus")
# # dataset = load_dataset("rag-datasets/rag-mini-wikipedia", "text-corpus")
# for split in dataset:
#     df = dataset[split].to_pandas()
#     output_file = f"DATASET/{dataset_name.replace('rag-datasets/', '')}/{split}.csv"
#     df.to_csv(output_file, index=False)
#     print(f"Saved {output_file} with shape {df.shape}")

CONFIGS = {
    "mandarjoshi/trivia_qa":        "rc.wikipedia",   # evidence pages for RAG
    "hotpotqa/hotpot_qa":           "distractor",     # 2 gold + 8 distractor paras
    "framolfese/2WikiMultihopQA":   None,             # HotpotQA-schema repackage
    # "nq_open":                      None,             # question+answers only, NO corpus
}

for name, cfg in CONFIGS.items():
    ds = load_dataset(name, cfg) if cfg else load_dataset(name)
    # ds = load_dataset(name, cfg, trust_remote_code=True)  # if it errors, add this
    for split in ds:
        out = f"DATASET/{name.split('/')[-1]}/{split}.parquet"
        os.makedirs(os.path.dirname(out), exist_ok=True)
        ds[split].to_parquet(out)        # parquet, not csv
        print(f"Saved {out} with shape {ds[split].shape}")
        