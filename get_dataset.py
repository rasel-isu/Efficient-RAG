
import pandas as pd
from datasets import load_dataset
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
import textwrap
import os
from datasets import load_dataset

# dataset = load_dataset("rag-datasets/rag-mini-wikipedia", "question-answer")
# for split in dataset:
#     df = dataset[split].to_pandas()
#     output_file = f"DATASET/rag-mini-wikipedia/{split}.csv"
#     df.to_csv(output_file, index=False)
#     print(f"Saved {output_file} with shape {df.shape}")

# dataset = load_dataset("rag-datasets/rag-mini-wikipedia", "text-corpus")
# for split in dataset:
#     df = dataset[split].to_pandas()
#     output_file = f"DATASET/rag-mini-wikipedia/{split}.csv"
#     df.to_csv(output_file, index=False)
#     print(f"Saved {output_file} with shape {df.shape}")

