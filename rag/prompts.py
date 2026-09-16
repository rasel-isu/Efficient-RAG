"""
Baseline-faithful prompt construction.

RECOMP (Appendix Table 13) prompts Flan-UL2 with five in-context Q/A pairs drawn
from the training split - no retrieved documents attached to the examples -
followed by the retrieved documents, the question, and "Answer:".

Using a hand-written format instruction instead (as this project did) breaks
badly on datasets that are not yes/no heavy: a small reader over-applies the
yes/no branch and answers "yes" to "Who is the mother of the director of ...".
Few-shot examples teach the output format by demonstration instead, which is
both more robust and what the baseline actually does.
"""

import random
from typing import List, Optional, Tuple

import pandas as pd

from rag.datasets import Passage

N_SHOT = 5          # RECOMP uses five in-context examples
_CACHE = {}


def _train_pairs(dataset: str) -> List[Tuple[str, str]]:
    """(question, answer) pairs from each dataset's TRAIN split, so in-context
    examples never leak from the evaluation split."""
    if dataset in _CACHE:
        return _CACHE[dataset]

    if dataset == "hotpotqa":
        df = pd.read_parquet("DATASET/hotpot_qa/train.parquet", columns=["question", "answer"])
        pairs = list(zip(df["question"], df["answer"]))
    elif dataset == "2wiki":
        df = pd.read_parquet("DATASET/2WikiMultihopQA/train.parquet", columns=["question", "answer"])
        pairs = list(zip(df["question"], df["answer"]))
    elif dataset == "musique":
        df = pd.read_parquet("DATASET/musique/train.parquet", columns=["question", "answer"])
        pairs = list(zip(df["question"], df["answer"]))
    elif dataset == "nq_open":
        df = pd.read_parquet("DATASET/nq_open/train.parquet", columns=["question", "answer"])
        pairs = [(q, a[0]) for q, a in zip(df["question"], df["answer"]) if len(a)]
    elif dataset == "triviaqa":
        df = pd.read_parquet("DATASET/trivia_qa/train.parquet", columns=["question", "answer"])
        pairs = [(q, a["value"]) for q, a in zip(df["question"], df["answer"])]
    elif dataset == "rag_mini":
        df = pd.read_csv("DATASET/rag-mini-wikipedia/test.csv")
        # rag-mini ships no train split; take examples from the tail so they do
        # not overlap the evaluated prefix
        pairs = list(zip(df["question"].tail(200), df["answer"].tail(200)))
    else:
        raise ValueError(f"no train pairs for {dataset}")

    pairs = [(str(q).strip(), str(a).strip()) for q, a in pairs if str(a).strip()]
    _CACHE[dataset] = pairs
    return pairs


def few_shot_block(dataset: str, n: int = N_SHOT, seed: int = 42) -> str:
    """Fixed across a run so every condition sees identical in-context examples."""
    pairs = _train_pairs(dataset)
    picked = random.Random(seed).sample(pairs, min(n, len(pairs)))
    return "\n".join(f"{q} Answer: {a}" for q, a in picked)


def order_passages_ascending(ranked: List[Tuple[Passage, float]]):
    """RECOMP: 'concatenate retrieved documents in ascending order of retrieval
    score, with the highest scored one closest to the question'."""
    return sorted(ranked, key=lambda x: x[1])


def build_prompt(dataset: str, question: str, context: str,
                 n_shot: int = N_SHOT, seed: int = 42,
                 instruction: Optional[str] = None) -> str:
    """RECOMP-style prompt. `instruction` is only for datasets where a format
    directive genuinely helps (rag-mini is ~38% yes/no); leave it None
    elsewhere."""
    parts = []
    if n_shot:
        parts.append(few_shot_block(dataset, n_shot, seed))
    if instruction:
        parts.append(instruction)
    parts.append(context)
    parts.append(question)
    parts.append("Answer:")
    return "\n".join(p for p in parts if p)


def clean_fewshot_answer(text: str, n_shot: int = N_SHOT) -> str:
    """Take only the first line of a few-shot continuation.

    The in-context block trains the model to emit `<question> Answer: <answer>`
    repeatedly, so after answering it happily invents the next question:
        'December 19, 1972\\nWhat is the largest planet...'
    Everything past the first newline is continuation, not answer. The untrimmed
    string is still stored as `rag_answer_raw`.
    """
    text = text.strip()
    if n_shot:
        text = text.split("\n")[0].strip()
        # a continuation can also restart inline as "... Answer: ..."
        if " Answer:" in text:
            text = text.split(" Answer:")[0].strip()
    return text
