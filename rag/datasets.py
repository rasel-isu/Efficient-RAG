"""
Unified loaders for the QA benchmarks used by the baseline papers
(RECOMP: NQ/TriviaQA/HotpotQA; CompAct: + 2WikiMultihopQA/MuSiQue).

Every loader returns a list of `Example`, which separates gold evidence from
distractor evidence. That separation is what lets us treat retrieval precision
as an experimental variable: `Example.passages(n_distractors=k)` builds a
candidate pool with a controlled amount of noise, from k=0 (oracle retrieval)
up to all available distractors (the noisy regime prior work evaluates in).
"""

from dataclasses import dataclass, field
from typing import List, Optional
import random

import pandas as pd


SAMPLE_SEED = 42        # fixed so every condition evaluates the same questions


def subsample(df, n, seed: int = SAMPLE_SEED):
    """Draw a reproducible random sample of `n` rows.

    Taking the first n rows instead is not safe: MuSiQue's validation split is
    grouped by hop count, so `df.iloc[:500]` returns 100% 2-hop questions where
    the full split is 52% 2-hop / 31% 3-hop / 17% 4-hop. RECOMP's protocol is a
    random sample, and a fixed seed keeps it identical across conditions so all
    comparisons stay paired.
    """
    if not n or n >= len(df):
        return df.reset_index(drop=True)
    return df.sample(n=n, random_state=seed).sort_index().reset_index(drop=True)


@dataclass
class Passage:
    title: str
    text: str
    is_gold: bool = False

    def render(self) -> str:
        return f"{self.title}\n{self.text}" if self.title else self.text


@dataclass
class Example:
    id: str
    question: str
    answers: List[str]
    gold: List[Passage] = field(default_factory=list)
    distractors: List[Passage] = field(default_factory=list)
    supporting_facts: List[str] = field(default_factory=list)
    question_type: Optional[str] = None
    dataset: str = ""

    def passages(self, n_distractors: Optional[int] = None, seed: int = 0) -> List[Passage]:
        """Candidate pool: all gold + `n_distractors` distractors, shuffled.
        n_distractors=None means use every distractor the dataset provides."""
        pool = list(self.distractors)
        if n_distractors is not None:
            rng = random.Random(seed)
            pool = rng.sample(pool, min(n_distractors, len(pool)))
        out = list(self.gold) + pool
        random.Random(seed).shuffle(out)
        return out


def _hotpot_style(path: str, dataset: str, n: Optional[int] = None,
                  seed: int = SAMPLE_SEED) -> List[Example]:
    """HotpotQA and 2WikiMultihopQA share a schema: `context` holds titles +
    per-title sentence lists, `supporting_facts` indexes into them."""
    df = subsample(pd.read_parquet(path), n, seed)
    examples = []
    for _, row in df.iterrows():
        ctx = row["context"]
        titles = list(ctx["title"])
        sents = [list(s) for s in ctx["sentences"]]
        by_title = dict(zip(titles, sents))

        sf = row["supporting_facts"]
        gold_titles = set(sf["title"])
        # resolve supporting facts down to the actual gold sentences
        sf_sentences = []
        for t, sid in zip(sf["title"], sf["sent_id"]):
            s = by_title.get(t)
            if s is not None and 0 <= sid < len(s):
                sf_sentences.append(s[sid].strip())

        gold, distractors = [], []
        for t, s in zip(titles, sents):
            p = Passage(title=t, text=" ".join(s).strip(), is_gold=t in gold_titles)
            (gold if p.is_gold else distractors).append(p)

        examples.append(Example(
            id=str(row["id"]), question=row["question"], answers=[str(row["answer"])],
            gold=gold, distractors=distractors, supporting_facts=sf_sentences,
            question_type=row.get("type"), dataset=dataset,
        ))
    return examples


def load_hotpotqa(split="validation", n=None, seed=SAMPLE_SEED) -> List[Example]:
    return _hotpot_style(f"DATASET/hotpot_qa/{split}.parquet", "hotpotqa", n, seed)


def load_2wiki(split="validation", n=None, seed=SAMPLE_SEED) -> List[Example]:
    return _hotpot_style(f"DATASET/2WikiMultihopQA/{split}.parquet", "2wikimultihopqa", n, seed)


def load_triviaqa(split="validation", n=None, chunk_words=100,
                  seed=SAMPLE_SEED) -> List[Example]:
    """TriviaQA rc.wikipedia: gold evidence is the question's entity pages.
    Articles are long, so they are split into ~100-word passages to match the
    100-word chunking RECOMP uses on its Wikipedia corpus."""
    df = pd.read_parquet(f"DATASET/trivia_qa/{split}.parquet")
    # drop evidence-less rows first: sampling then dropping would return < n
    df = df[[len(ep["wiki_context"]) > 0 for ep in df["entity_pages"]]]
    df = subsample(df, n, seed)
    examples = []
    for _, row in df.iterrows():
        ep = row["entity_pages"]
        gold = []
        for title, ctx in zip(ep["title"], ep["wiki_context"]):
            words = ctx.split()
            for i in range(0, len(words), chunk_words):
                gold.append(Passage(title=str(title),
                                    text=" ".join(words[i:i + chunk_words]),
                                    is_gold=True))
        if not gold:
            continue  # some rows carry no wiki evidence
        ans = row["answer"]
        aliases = [str(a) for a in ans["aliases"]] or [str(ans["value"])]
        examples.append(Example(
            id=str(row["question_id"]), question=row["question"],
            answers=[str(ans["value"])] + [a for a in aliases if a != str(ans["value"])],
            gold=gold, distractors=[], supporting_facts=[],
            dataset="triviaqa",
        ))
    return examples


def load_musique(split="validation", n=None, seed=SAMPLE_SEED) -> List[Example]:
    """MuSiQue: 20 paragraphs per question, typically 2 supporting. The larger
    distractor pool (18 vs HotpotQA's 8) gives the widest retrieval-noise range
    of any dataset here.

    Support is annotated at paragraph rather than sentence level, so
    `supporting_facts` holds gold paragraph text; the atomic per-hop answers are
    kept separately in `hop_answers` as a stricter retention signal."""
    df = subsample(pd.read_parquet(f"DATASET/musique/{split}.parquet"), n, seed)
    examples = []
    for _, row in df.iterrows():
        # is_supporting round-trips through parquet as a string
        def _is_gold(v):
            return str(v).lower() == "true"

        gold, distractors, sf = [], [], []
        for para in row["paragraphs"]:
            p = Passage(title=str(para["title"]), text=str(para["paragraph_text"]),
                        is_gold=_is_gold(para["is_supporting"]))
            if p.is_gold:
                gold.append(p)
                sf.append(p.text)
            else:
                distractors.append(p)

        aliases = [str(a) for a in (row["answer_aliases"] if row["answer_aliases"] is not None else [])]
        ex = Example(
            id=str(row["id"]), question=row["question"],
            answers=[str(row["answer"])] + [a for a in aliases if a != str(row["answer"])],
            gold=gold, distractors=distractors, supporting_facts=sf,
            question_type=f"{len(row['question_decomposition'])}-hop", dataset="musique",
        )
        ex.hop_answers = [str(h["answer"]) for h in row["question_decomposition"]]
        examples.append(ex)
    return examples


def load_nq_open(split="validation", n=None, seed=SAMPLE_SEED) -> List[Example]:
    """NQ-Open ships questions and answers only - no corpus. Returned with empty
    evidence; a corpus must be attached before this is usable for RAG."""
    df = subsample(pd.read_parquet(f"DATASET/nq_open/{split}.parquet"), n, seed)
    return [Example(id=f"nq-{i}", question=r["question"],
                    answers=[str(a) for a in r["answer"]], dataset="nq_open")
            for i, r in df.iterrows()]


LOADERS = {
    "hotpotqa": load_hotpotqa,
    "2wiki": load_2wiki,
    "triviaqa": load_triviaqa,
    "musique": load_musique,
    "nq_open": load_nq_open,
}


def inject_cross_question_distractors(examples: List[Example], n: int = 8,
                                      seed: int = 0) -> List[Example]:
    """Give datasets without native distractors (TriviaQA) a comparable noise
    axis by sampling evidence passages from *other* questions.

    TriviaQA annotates evidence at the article level, so every chunk of a gold
    entity page is labelled gold even though most chunks are irrelevant to the
    question. Injecting passages from unrelated questions is what makes the
    retrieval-precision sweep comparable to HotpotQA/2Wiki distractor pools.
    """
    rng = random.Random(seed)
    pool = [(i, p) for i, ex in enumerate(examples) for p in ex.gold]
    if not pool:
        return examples
    for i, ex in enumerate(examples):
        picked, guard = [], 0
        while len(picked) < n and guard < n * 50:
            j, p = pool[rng.randrange(len(pool))]
            guard += 1
            if j != i:
                picked.append(Passage(title=p.title, text=p.text, is_gold=False))
        ex.distractors = picked
    return examples
