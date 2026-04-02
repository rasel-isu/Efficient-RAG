
import pandas as pd
from datasets import load_dataset
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

