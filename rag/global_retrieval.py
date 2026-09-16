"""
Retrieval over the DPR Wikipedia 2018 corpus (21M passages), as opposed to the
per-question candidate pools in rag.local_retrieval.

This is the baseline-faithful path: RECOMP and CompAct both retrieve from this
corpus with Contriever-MSMARCO. Contriever scores by DOT PRODUCT over masked
mean-pooled embeddings and is deliberately not L2-normalised - normalising it
would silently change the ranking away from the published setup.
"""

import os
import shutil
import time

import numpy as np
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModel

from rag.datasets import Passage
from rag.local_retrieval import LocalRAG
from rag.encoders import ENCODERS, mean_pool, normalizes


# Lustre delivers only a few MB/s for the mmap access pattern this load needs
# (random page faults over 32 GB), so the index is staged once onto local NVMe.
LOCAL_CACHE = os.environ.get("RAG_INDEX_CACHE", "/tmp/efficient-rag-index")


def _stage_locally(path: str) -> str:
    """Copy `path` to local scratch once; return whichever copy to read from."""
    try:
        os.makedirs(LOCAL_CACHE, exist_ok=True)
    except OSError:
        return path
    dst = os.path.join(LOCAL_CACHE, path.replace("/", "_"))
    src_size = os.path.getsize(path)
    if os.path.exists(dst) and os.path.getsize(dst) == src_size:
        print(f"[index] using local copy {dst}", flush=True)
        return dst
    if shutil.disk_usage(LOCAL_CACHE).free < src_size * 1.1:
        print(f"[index] insufficient local scratch; reading {path} directly", flush=True)
        return path
    # Unique temp name per process: with one worker per GPU on a fresh node,
    # several may stage at once and a shared ".part" file would corrupt.
    tmp = f"{dst}.{os.getpid()}.part"
    print(f"[index] staging {src_size/1e9:.1f} GB -> {dst} (one time)", flush=True)
    t0 = time.perf_counter()
    try:
        with open(path, "rb", buffering=0) as fsrc, open(tmp, "wb", buffering=0) as fdst:
            shutil.copyfileobj(fsrc, fdst, length=64 * 1024 * 1024)
        os.replace(tmp, dst)           # atomic; last writer wins, contents identical
    except OSError as e:
        print(f"[index] staging failed ({e}); reading {path} directly", flush=True)
        if os.path.exists(tmp):
            os.unlink(tmp)
        return path
    print(f"[index] staged in {time.perf_counter()-t0:.0f}s", flush=True)
    return dst


class GlobalRAG(LocalRAG):

    def __init__(self, *a, index_dir: str = "DATASET/wiki_dpr/index_contriever",
                 encoder: str = "contriever", prefetch: int = 100, **kw):
        super().__init__(*a, **kw)
        self.encoder_name = encoder
        self.passages = pd.read_parquet(f"{index_dir}/passages.parquet")
        emb_path = _stage_locally(f"{index_dir}/embeddings.fp16.npy")
        emb = np.load(emb_path, mmap_mode="r")
        if len(emb) != len(self.passages):
            raise ValueError(f"index/passage mismatch: {len(emb)} vs {len(self.passages)}")
        dev = "cuda" if torch.cuda.is_available() else "cpu"
        # 21M x 768 fp16 is ~32 GB - fits an 80 GB A100, so search stays exact.
        # Stream it in chunks: materialising the whole array in host RAM first
        # (np.ascontiguousarray on the memmap) turns the load into random page
        # faults and takes hours on a parallel filesystem.
        self.emb = torch.empty(emb.shape, dtype=torch.float16, device=dev)
        step = 1_000_000
        for i in range(0, len(emb), step):
            self.emb[i:i + step] = torch.from_numpy(
                np.array(emb[i:i + step], dtype=np.float16)).to(dev, non_blocking=True)
        self.q_tok = AutoTokenizer.from_pretrained(ENCODERS[encoder])
        self.q_model = AutoModel.from_pretrained(
            ENCODERS[encoder], torch_dtype=torch.float16).to(dev).eval()
        self.prefetch = prefetch

    def _embed_query(self, question: str) -> torch.Tensor:
        if normalizes(self.encoder_name):
            question = f"Represent this sentence for searching relevant passages: {question}"
        enc = self.q_tok([question], return_tensors="pt", truncation=True,
                         max_length=256).to(self.emb.device)
        with torch.inference_mode():
            v = mean_pool(self.q_model(**enc).last_hidden_state, enc["attention_mask"])
        if normalizes(self.encoder_name):
            v = torch.nn.functional.normalize(v, dim=-1)
        return v.to(self.emb.dtype)[0]

    def retrieve(self, question: str, passages=None):
        """`passages` is ignored - NQ and the baseline-faithful multi-hop runs
        retrieve from the whole corpus."""
        sims = (self.emb @ self._embed_query(question)).float()
        idx = torch.topk(sims, k=min(self.prefetch, sims.numel())).indices.tolist()
        rows = self.passages.iloc[idx]
        cand = [Passage(title=str(t), text=str(x), is_gold=False)
                for t, x in zip(rows["title"], rows["text"])]

        if self.reranker is None:                 # raw dense retrieval (baseline)
            return [(c, float(sims[i])) for c, i in zip(cand, idx)][:self.top_k]
        scores = self.reranker.predict([(question, p.render()) for p in cand],
                                       show_progress_bar=False)
        order = sorted(range(len(cand)), key=lambda i: -scores[i])[:self.top_k]
        return [(cand[i], float(scores[i])) for i in order]
