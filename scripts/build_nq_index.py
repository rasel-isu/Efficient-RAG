"""
Embed the BEIR NQ corpus (2.68M passages) with the same bge-base encoder the
rest of the pipeline uses.

NQ-Open ships questions and answers only, so unlike HotpotQA/2Wiki/MuSiQue there
is no per-question candidate pool - it needs a real corpus and real retrieval.
That makes NQ the "realistic retrieval" condition rather than part of the
synthetic noise sweep (it has no gold-passage labels to dial noise against).

Output: fp16 embeddings memmap + the passage table. 2.68M x 768 fp16 is ~4.1 GB,
which fits in A100 memory, so retrieval is exact matmul - no ANN index needed.
"""
import argparse, os
import numpy as np
import pandas as pd
import torch
from sentence_transformers import SentenceTransformer

EMBED_MODEL = "BAAI/bge-base-en-v1.5"
OUT_DIR = "DATASET/nq_open/index"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch-size", type=int, default=512)
    ap.add_argument("--limit", type=int, default=None, help="debug: embed only N passages")
    args = ap.parse_args()

    os.makedirs(OUT_DIR, exist_ok=True)
    df = pd.read_parquet("DATASET/nq_open/corpus.parquet")
    if args.limit:
        df = df.iloc[:args.limit]
    n = len(df)
    print(f"corpus: {n:,} passages")

    texts = (df["title"].fillna("") + "\n" + df["text"].fillna("")).tolist()
    model = SentenceTransformer(EMBED_MODEL, device="cuda" if torch.cuda.is_available() else "cpu")
    dim = model.get_sentence_embedding_dimension()

    emb_path = f"{OUT_DIR}/embeddings.fp16.npy"
    mm = np.lib.format.open_memmap(emb_path, mode="w+", dtype=np.float16, shape=(n, dim))

    step = 50_000
    for start in range(0, n, step):
        chunk = texts[start:start + step]
        vecs = model.encode(chunk, batch_size=args.batch_size, normalize_embeddings=True,
                            convert_to_numpy=True, show_progress_bar=False)
        mm[start:start + len(chunk)] = vecs.astype(np.float16)
        print(f"  {min(start+step, n):>9,}/{n:,}", flush=True)
    mm.flush()

    df[["_id", "title", "text"]].to_parquet(f"{OUT_DIR}/passages.parquet")
    print(f"wrote {emb_path} ({os.path.getsize(emb_path)/1e9:.2f} GB) and passages.parquet")


if __name__ == "__main__":
    main()
