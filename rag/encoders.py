"""Shared encoder definitions, so the index builder and the query-time retriever
cannot drift apart in pooling or normalisation."""

import torch

ENCODERS = {
    "contriever": "facebook/contriever-msmarco",   # RECOMP + CompAct baseline
    "bge":        "BAAI/bge-base-en-v1.5",         # ours
}


def mean_pool(hidden, mask):
    """Masked mean pooling over the last hidden state (Contriever's pooling)."""
    hidden = hidden.masked_fill(~mask[..., None].bool(), 0.0)
    return hidden.sum(dim=1) / mask.sum(dim=1)[..., None]


def normalizes(encoder: str) -> bool:
    """bge is trained for cosine similarity; Contriever scores by dot product
    and must NOT be normalised or ranking diverges from the published setup."""
    return encoder == "bge"
