"""
Per-question retrieval over a small candidate pool (HotpotQA / 2Wiki distractor
setting, TriviaQA entity pages).

The Weaviate-backed `Indexing` path builds one index over a shared corpus, which
is right for rag-mini-wikipedia but unusable here: these datasets carry a
different candidate pool per question, and rebuilding a Weaviate collection per
question would dominate runtime. This module keeps the same retrieval stack -
BAAI/bge-base-en-v1.5 embeddings, BAAI/bge-reranker-base cross-encoder - but
runs it in memory over the handful of passages belonging to one question.

Compression reuses the exact functions from rag.retrieval, so the compressed
condition is identical across every dataset.
"""

import time
from typing import List, Optional

import torch
from llama_index.core import Settings
from sentence_transformers import SentenceTransformer, CrossEncoder
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

from rag.datasets import Passage
from rag.prompts import build_prompt, order_passages_ascending, clean_fewshot_answer
from rag.retrieval import (
    _device, extract_relevant_sentences, summarize_for_query_with_chunks,
)

EMBED_MODEL = "BAAI/bge-base-en-v1.5"
RERANK_MODEL = "BAAI/bge-reranker-base"


class LocalRAG:

    def __init__(self, token_counter, summ_model_name: Optional[str] = None,
                 tokenizer=None, top_k: int = 5, use_reranker: bool = True):
        self.embedder = SentenceTransformer(EMBED_MODEL, device=str(_device))
        # Reranking is THE contribution variable: RECOMP, CompAct and the
        # scaling paper all retrieve without a cross-encoder, so reproducing
        # them requires use_reranker=False.
        self.reranker = CrossEncoder(RERANK_MODEL, device=str(_device)) if use_reranker else None
        self.token_counter = token_counter
        self.tokenizer = tokenizer
        self.top_k = top_k
        self.summ_model = None
        if summ_model_name:
            self.summ_tokenizer = AutoTokenizer.from_pretrained(summ_model_name)
            self.summ_model = AutoModelForSeq2SeqLM.from_pretrained(summ_model_name).to(_device)
            self.summ_model.eval()

    def _n_tokens(self, text: str) -> int:
        if self.tokenizer is None:
            return len(text.split())
        return len(self.tokenizer(text))

    def retrieve(self, question: str, passages: List[Passage]):
        """Dense top-k, then cross-encoder rerank - the same two stages the
        Weaviate path uses, so retrieval quality is comparable across datasets."""
        texts = [p.render() for p in passages]
        if not texts:
            return []
        # bge asks for this prefix on the query side only
        q_emb = self.embedder.encode(
            [f"Represent this sentence for searching relevant passages: {question}"],
            normalize_embeddings=True, show_progress_bar=False)
        p_emb = self.embedder.encode(texts, normalize_embeddings=True,
                                     batch_size=64, show_progress_bar=False)
        sims = (p_emb @ q_emb[0])
        k = min(self.top_k, len(texts))
        top = sims.argsort()[::-1][:k]
        cand = [passages[i] for i in top]

        if self.reranker is None:
            return [(passages[i], float(sims[i])) for i in top]
        scores = self.reranker.predict([(question, p.render()) for p in cand],
                                       show_progress_bar=False)
        order = sorted(range(len(cand)), key=lambda i: -scores[i])
        return [(cand[i], float(scores[i])) for i in order]

    def _truncate(self, text: str, budget: int) -> str:
        """Keep the first `budget` tokens, measured with the generator's tokenizer."""
        if budget <= 0:
            return text
        words = text.split()
        # binary search on word count to hit the token budget without decoding
        lo, hi = 0, len(words)
        while lo < hi:
            mid = (lo + hi + 1) // 2
            if self._n_tokens(" ".join(words[:mid])) <= budget:
                lo = mid
            else:
                hi = mid - 1
        return " ".join(words[:lo])

    def answer(self, question: str, passages: List[Passage],
               human_prompt: str = None, use_summarizer: bool = False,
               use_keyword_filtering: bool = False, truncate_to: int = 0,
               dataset: str = "hotpotqa", n_shot: int = 5):
        t0 = time.perf_counter()
        ranked = self.retrieve(question, passages)
        t_retrieval = (time.perf_counter() - t0) * 1000
        # RECOMP concatenates documents in ASCENDING retrieval score, so the
        # best-scoring passage sits closest to the question
        ranked = order_passages_ascending(ranked)

        raw_texts = [p.render() for p, _ in ranked]
        retrieved_chunks = [{"title": p.title, "text": p.render(), "score": s,
                             "is_gold": p.is_gold} for p, s in ranked]
        # Measure the uncompressed context in the SAME rendering the compressed
        # one uses (snippet headers included). Counting raw text against
        # header-bearing compressed text made the uncompressed baseline report
        # 0.92x instead of 1.00x, understating every compression rate.
        raw_rendered = "\n\n".join(
            f"[Snippet {i+1} | score={sc:.3f}]\n{t}"
            for i, (t, (_, sc)) in enumerate(zip(raw_texts, ranked)))
        retrieved_context_tokens = self._n_tokens(raw_rendered)

        # Budget the TOTAL rendered context, not the passage text alone: with
        # top-30 the "[Snippet N | score=...]" headers cost ~300 tokens, which
        # would otherwise push truncate far past the compressor it is matched to.
        per_passage_budget = 0
        if truncate_to:
            header_cost = sum(
                self._n_tokens(f"[Snippet {i+1} | score={sc:.3f}]\n")
                for i, (_, sc) in enumerate(ranked))
            per_passage_budget = max(
                1, (truncate_to - header_cost) // max(len(raw_texts), 1))

        snippets, t_filter, t_summarize = [], 0.0, 0.0
        for i, raw_text in enumerate(raw_texts):
            if use_keyword_filtering:
                tf = time.perf_counter()
                relevant = extract_relevant_sentences(raw_text, question, max_sentences=4)
                t_filter += (time.perf_counter() - tf) * 1000
            elif truncate_to:
                tf = time.perf_counter()
                relevant = self._truncate(raw_text, per_passage_budget)
                t_filter += (time.perf_counter() - tf) * 1000
            else:
                relevant = raw_text

            if use_summarizer:
                ts = time.perf_counter()
                relevant = summarize_for_query_with_chunks(
                    relevant, question, self.summ_tokenizer, self.summ_model,
                    max_new_tokens=1000)
                t_summarize += (time.perf_counter() - ts) * 1000

            snippets.append(f"[Snippet {i+1} | score={ranked[i][1]:.3f}]\n{relevant}")

        context = "\n\n".join(snippets)
        compressed_context_tokens = self._n_tokens(context)

        final_prompt = build_prompt(dataset, question, context,
                                    n_shot=n_shot, instruction=human_prompt)

        tg = time.perf_counter()
        resp = Settings.llm.complete(final_prompt)
        t_generate = (time.perf_counter() - tg) * 1000

        last = self.token_counter.llm_token_counts[-1]
        return clean_fewshot_answer(resp.text, n_shot), {
            'prompt_token': last.prompt_token_count,
            'completion_token': last.completion_token_count,
            'total_token': last.total_token_count,
            'rag_answer_raw': resp.text,
            'compressed_context': context,
            'compressed_context_tokens': compressed_context_tokens,
            'retrieved_context_tokens': retrieved_context_tokens,
            'retrieved_chunks': retrieved_chunks,
            'n_gold_retrieved': sum(1 for c in retrieved_chunks if c["is_gold"]),
            't_retrieval_ms': t_retrieval,
            't_filter_ms': t_filter,
            't_summarize_ms': t_summarize,
            't_generate_ms': t_generate,
            't_end_to_end_ms': t_retrieval + t_filter + t_summarize + t_generate,
        }
