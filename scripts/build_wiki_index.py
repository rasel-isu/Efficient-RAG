"""
Embed the DPR Wikipedia 2018 corpus (21M x 100-word passages) for baseline-
faithful retrieval.

RECOMP and CompAct both retrieve with Contriever-MSMARCO over this corpus, so
reproducing their numbers requires this exact retriever/corpus pair. The bge
encoder used elsewhere in this project is the *contribution* condition and gets
its own pass (--encoder bge).

Contriever pools by masked mean over the last hidden state and scores by dot
product - it is NOT a sentence-transformers model and must not be L2-normalised,
or the scores stop matching the published setup.
"""
import argparse, glob, os, sys
import numpy as np
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModel

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from rag.encoders import ENCODERS, mean_pool, normalizes


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--encoder", default="contriever", choices=list(ENCODERS))
    ap.add_argument("--batch-size", type=int, default=1024)
    ap.add_argument("--max-length", type=int, default=256)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--corpus-glob", default="DATASET/wiki_dpr/corpus-*.parquet")
    ap.add_argument("--out-dir", default=None)
    args = ap.parse_args()

    out_dir = args.out_dir or f"DATASET/wiki_dpr/index_{args.encoder}"
    os.makedirs(out_dir, exist_ok=True)

    files = sorted(glob.glob(args.corpus_glob))
    if not files:
        raise SystemExit(f"no corpus shards matching {args.corpus_glob}")
    df = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    if args.limit:
        df = df.iloc[:args.limit]
    n = len(df)
    print(f"corpus: {n:,} passages from {len(files)} shard(s)", flush=True)

    name = ENCODERS[args.encoder]
    tok = AutoTokenizer.from_pretrained(name)
    model = AutoModel.from_pretrained(name, torch_dtype=torch.float16).cuda().eval()
    dim = model.config.hidden_size

    texts = (df["title"].fillna("") + ". " + df["text"].fillna("")).tolist()
    mm = np.lib.format.open_memmap(f"{out_dir}/embeddings.fp16.npy", mode="w+",
                                   dtype=np.float16, shape=(n, dim))

    done = 0
    with torch.inference_mode():
        for start in range(0, n, args.batch_size):
            batch = texts[start:start + args.batch_size]
            enc = tok(batch, padding=True, truncation=True,
                      max_length=args.max_length, return_tensors="pt").to("cuda")
            out = model(**enc).last_hidden_state
            vecs = mean_pool(out, enc["attention_mask"])
            if normalizes(args.encoder):
                vecs = torch.nn.functional.normalize(vecs, dim=-1)
            mm[start:start + len(batch)] = vecs.to(torch.float16).cpu().numpy()
            done += len(batch)
            if (start // args.batch_size) % 200 == 0:
                print(f"  {done:>10,}/{n:,}", flush=True)
    mm.flush()
    df[["id", "title", "text"]].to_parquet(f"{out_dir}/passages.parquet")
    print(f"done -> {out_dir} ({os.path.getsize(out_dir+'/embeddings.fp16.npy')/1e9:.1f} GB)")


if __name__ == "__main__":
    main()
