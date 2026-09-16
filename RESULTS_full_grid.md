# Full Grid — Token Efficiency Analysis

Sampling: **42** random seed, identical question set in every cell.

Contribution metric is **token cost**: how far the prompt can be compressed before accuracy becomes unacceptable. Accuracy is reported as retention against the uncompressed baseline in the same retrieval setting.

## How to read this report

| Term | Meaning |
|---|---|
| **Ours** | Keyword filter + FLAN-T5 compression of the retrieved context, Llama-3.2-3B reader |
| Comp. | Compression ratio: context tokens before ÷ after |
| Token reduction | Share of context tokens removed; higher is better |
| EM retention | Exact match as a share of the **same system's own** uncompressed run. In Table 1 that is each paper's own no-compression row, which is what makes the numbers readable across readers of different sizes; elsewhere it is our uncompressed cell in the same rerank setting |
| Δ | **Ours minus the row**, so a positive Δ means the proposed method wins. Δ on reduction and retention is in percentage points |
| rerank | bge cross-encoder reranking of the retrieved passages, on or off |

**Table 1** compares the method against published compressors, **Table 2** against its own ablations, **Table 3** against in-pipeline controls (no compression, and truncation at a matched token budget). Everything after them is supporting detail: confidence intervals, significance tests, supporting-fact retention and latency.

## Table 1 — Ours vs published compressors

Published rows are transcribed from CompAct (EMNLP 2024) Table 2 — reader LLaMA3-8B, Contriever top-30, no reranking. That grid shares our retriever, corpus and depth, but uses a larger reader (LLaMA3-8B vs Llama-3.2-3B) and no reranking - so our rerank-off cell is the matched setting and is the reference row. Δ is *ours minus the row*, so a positive Δ means our method is ahead.

Because the readers differ, raw EM is not a fair column. **EM retention** is: each system's EM as a share of its own uncompressed row (`Raw Document` for the published systems, our uncompressed cell for ours). It measures what the compressor costs its own reader, which is what a compression paper is claiming.

**Summary — mean over the datasets each system reports**

| System | Reader | Mean comp. | Mean EM | Mean EM retention | n datasets |
|---|---|---|---|---|---|
| **Ours: Keyword filter + FLAN-T5** | Llama-3.2-3B | 8.1× | 30.0 | 85% | 5 |
| Ours (rerank on) | Llama-3.2-3B | 8.2× | 31.8 | 87% | 5 |
| Ours, uncompressed | Llama-3.2-3B | 1.0× | 35.2 | 100% | 5 |
| Raw Document | LLaMA3-8B | 1.0× | 33.8 | 100% | 5 |
| AutoCompressors | LLaMA3-8B | 35.0× | 22.8 | 64% | 5 |
| LongLLMLingua | LLaMA3-8B | 3.4× | 30.0 | 87% | 5 |
| RECOMP (extractive) | LLaMA3-8B | 35.0× | 33.7 | 102% | 5 |
| CompAct | LLaMA3-8B | 46.8× | 35.8 | 114% | 5 |
| Oracle | LLaMA3-8B | 10.7× | 30.5 | 167% | 3 |

Oracle is the gold-supporting-documents upper bound, not a deployable system. Mean EM mixes datasets of very different difficulty and is only comparable within a reader; mean retention is comparable across readers.

**TriviaQA**

| System | Reader | Comp. | EM | ΔEM | F1 | ΔF1 | Retention | ΔRet |
|---|---|---|---|---|---|---|---|---|
| **Ours: Keyword filter + FLAN-T5** | Llama-3.2-3B | 8.9× | 68.6 | — | 73.6 | — | 90% | — |
| Ours (rerank on) | Llama-3.2-3B | 9.0× | 70.8 | -2.2 | 75.3 | -1.7 | 88% | +1 |
| Ours, uncompressed | Llama-3.2-3B | 1.0× | 76.6 | -8.0 | 82.2 | -8.7 | 100% | -10 |
| Raw Document | LLaMA3-8B | 1.0× | 68.9 | -0.3 | 77.1 | -3.5 | 100% | -10 |
| AutoCompressors | LLaMA3-8B | 34.5× | 55.3 | +13.3 | 64.3 | +9.3 | 80% | +9 |
| LongLLMLingua | LLaMA3-8B | 3.3× | 64.0 | +4.6 | 70.8 | +2.8 | 93% | -3 |
| RECOMP (extractive) | LLaMA3-8B | 39.2× | 67.6 | +1.0 | 74.1 | -0.5 | 98% | -9 |
| CompAct | LLaMA3-8B | 49.4× | 65.4 | +3.2 | 74.9 | -1.3 | 95% | -5 |

**NQ-Open**

| System | Reader | Comp. | EM | ΔEM | F1 | ΔF1 | Retention | ΔRet |
|---|---|---|---|---|---|---|---|---|
| **Ours: Keyword filter + FLAN-T5** | Llama-3.2-3B | 5.8× | 27.4 | — | 38.0 | — | 83% | — |
| Ours (rerank on) | Llama-3.2-3B | 6.0× | 29.4 | -2.0 | 37.5 | +0.5 | 85% | -3 |
| Ours, uncompressed | Llama-3.2-3B | 1.0× | 33.2 | -5.8 | 44.4 | -6.4 | 100% | -17 |
| Raw Document | LLaMA3-8B | 1.0× | 39.0 | -11.6 | 51.3 | -13.3 | 100% | -17 |
| AutoCompressors | LLaMA3-8B | 34.4× | 17.3 | +10.1 | 31.8 | +6.2 | 44% | +38 |
| LongLLMLingua | LLaMA3-8B | 3.5× | 27.7 | -0.3 | 40.6 | -2.6 | 71% | +12 |
| RECOMP (extractive) | LLaMA3-8B | 32.7× | 34.6 | -7.2 | 45.1 | -7.1 | 89% | -6 |
| CompAct | LLaMA3-8B | 48.5× | 38.4 | -11.0 | 50.0 | -12.0 | 98% | -16 |

**HotpotQA**

| System | Reader | Comp. | EM | ΔEM | F1 | ΔF1 | Retention | ΔRet |
|---|---|---|---|---|---|---|---|---|
| **Ours: Keyword filter + FLAN-T5** | Llama-3.2-3B | 8.5× | 22.6 | — | 31.0 | — | 75% | — |
| Ours (rerank on) | Llama-3.2-3B | 8.6× | 23.2 | -0.6 | 31.8 | -0.8 | 75% | +0 |
| Ours, uncompressed | Llama-3.2-3B | 1.0× | 30.2 | -7.6 | 40.1 | -9.1 | 100% | -25 |
| Raw Document | LLaMA3-8B | 1.0× | 29.4 | -6.8 | 40.3 | -9.3 | 100% | -25 |
| AutoCompressors | LLaMA3-8B | 35.4× | 18.4 | +4.2 | 28.4 | +2.6 | 63% | +12 |
| LongLLMLingua | LLaMA3-8B | 3.4× | 25.6 | -3.0 | 35.3 | -4.3 | 87% | -12 |
| RECOMP (extractive) | LLaMA3-8B | 34.3× | 29.7 | -7.1 | 39.9 | -8.9 | 101% | -26 |
| CompAct | LLaMA3-8B | 47.6× | 35.5 | -12.9 | 46.9 | -15.9 | 121% | -46 |
| Oracle | LLaMA3-8B | 10.8× | 39.9 | -17.3 | 51.2 | -20.2 | 136% | -61 |

**2WikiMultihopQA**

| System | Reader | Comp. | EM | ΔEM | F1 | ΔF1 | Retention | ΔRet |
|---|---|---|---|---|---|---|---|---|
| **Ours: Keyword filter + FLAN-T5** | Llama-3.2-3B | 9.8× | 25.8 | — | 30.6 | — | 87% | — |
| Ours (rerank on) | Llama-3.2-3B | 9.8× | 28.0 | -2.2 | 32.4 | -1.8 | 97% | -10 |
| Ours, uncompressed | Llama-3.2-3B | 1.0× | 29.8 | -4.0 | 36.4 | -5.8 | 100% | -13 |
| Raw Document | LLaMA3-8B | 1.0× | 25.4 | +0.4 | 31.2 | -0.6 | 100% | -13 |
| AutoCompressors | LLaMA3-8B | 36.2× | 19.0 | +6.8 | 24.5 | +6.1 | 75% | +12 |
| LongLLMLingua | LLaMA3-8B | 3.6× | 27.9 | -2.1 | 32.9 | -2.3 | 110% | -23 |
| RECOMP (extractive) | LLaMA3-8B | 35.9× | 29.9 | -4.1 | 34.9 | -4.3 | 118% | -31 |
| CompAct | LLaMA3-8B | 51.2× | 31.0 | -5.2 | 37.1 | -6.5 | 122% | -35 |
| Oracle | LLaMA3-8B | 11.0× | 37.4 | -11.6 | 43.2 | -12.6 | 147% | -61 |

**MuSiQue**

| System | Reader | Comp. | EM | ΔEM | F1 | ΔF1 | Retention | ΔRet |
|---|---|---|---|---|---|---|---|---|
| **Ours: Keyword filter + FLAN-T5** | Llama-3.2-3B | 7.5× | 5.6 | — | 13.6 | — | 93% | — |
| Ours (rerank on) | Llama-3.2-3B | 7.5× | 7.8 | -2.2 | 16.7 | -3.1 | 91% | +3 |
| Ours, uncompressed | Llama-3.2-3B | 1.0× | 6.0 | -0.4 | 14.0 | -0.5 | 100% | -7 |
| Raw Document | LLaMA3-8B | 1.0× | 6.5 | -0.9 | 15.6 | -2.0 | 100% | -7 |
| AutoCompressors | LLaMA3-8B | 34.7× | 3.9 | +1.7 | 11.9 | +1.7 | 60% | +33 |
| LongLLMLingua | LLaMA3-8B | 3.4× | 4.8 | +0.8 | 13.5 | +0.1 | 74% | +19 |
| RECOMP (extractive) | LLaMA3-8B | 32.7× | 6.7 | -1.1 | 15.7 | -2.1 | 103% | -10 |
| CompAct | LLaMA3-8B | 37.2× | 8.7 | -3.1 | 18.1 | -4.5 | 134% | -41 |
| Oracle | LLaMA3-8B | 10.3× | 14.2 | -8.6 | 23.6 | -10.0 | 218% | -125 |

### Closest published analogue — off-the-shelf summarisation

RECOMP (ICLR 2024) Table 2 — reader Flan-UL2 20B, Contriever top-5. A different reader *and* depth, so this is positioning rather than a like-for-like comparison. The `T5 (off-the-shelf)` row is an untrained summariser applied to retrieved passages - the same recipe as ours - and is the number this method has to be read against.

| Dataset | System | Reader | EM | F1 | EM retention |
|---|---|---|---|---|---|
| TriviaQA | **Ours: Keyword filter + FLAN-T5** | Llama-3.2-3B | 68.6 | 73.6 | 90% |
| TriviaQA | Top 5 documents | Flan-UL2 20B | 62.4 | 70.1 | 100% |
| TriviaQA | Top 1 document | Flan-UL2 20B | 57.8 | 64.9 | 93% |
| TriviaQA | no retrieval | Flan-UL2 20B | 49.3 | 54.9 | 79% |
| TriviaQA | T5 (off-the-shelf) | Flan-UL2 20B | 55.2 | 62.3 | 88% |
| TriviaQA | RECOMP abstractive | Flan-UL2 20B | 58.7 | 66.3 | 94% |
| TriviaQA | RECOMP extractive | Flan-UL2 20B | 59.0 | 65.3 | 95% |
| NQ-Open | **Ours: Keyword filter + FLAN-T5** | Llama-3.2-3B | 27.4 | 38.0 | 83% |
| NQ-Open | Top 5 documents | Flan-UL2 20B | 39.4 | 48.3 | 100% |
| NQ-Open | Top 1 document | Flan-UL2 20B | 33.1 | 41.5 | 84% |
| NQ-Open | no retrieval | Flan-UL2 20B | 22.0 | 29.4 | 56% |
| NQ-Open | T5 (off-the-shelf) | Flan-UL2 20B | 25.9 | 34.6 | 66% |
| NQ-Open | RECOMP abstractive | Flan-UL2 20B | 37.0 | 45.5 | 94% |
| NQ-Open | RECOMP extractive | Flan-UL2 20B | 36.6 | 44.2 | 93% |
| HotpotQA | **Ours: Keyword filter + FLAN-T5** | Llama-3.2-3B | 22.6 | 31.0 | 75% |
| HotpotQA | Top 5 documents | Flan-UL2 20B | 32.8 | 43.9 | 100% |
| HotpotQA | Top 1 document | Flan-UL2 20B | 28.8 | 40.6 | 88% |
| HotpotQA | no retrieval | Flan-UL2 20B | 17.8 | 26.1 | 54% |
| HotpotQA | T5 (off-the-shelf) | Flan-UL2 20B | 23.2 | 33.2 | 71% |
| HotpotQA | RECOMP abstractive | Flan-UL2 20B | 28.2 | 37.9 | 86% |
| HotpotQA | RECOMP extractive | Flan-UL2 20B | 30.4 | 40.1 | 93% |

## Table 2 — Ours vs our variants

Ablations of the proposed pipeline. The reference row is the full system (Keyword filter + FLAN-T5, rerank on); every Δ is that row minus the variant.

| Dataset | System | EM | ΔEM | F1 | ΔF1 | BERTScore | ΔBERT | Token redn | ΔRedn | Retention | ΔRet |
|---|---|---|---|---|---|---|---|---|---|---|---|
| TriviaQA | **Ours: Keyword filter + FLAN-T5** (rerank on, reference) | 70.8 | — | 75.3 | — | 92.23 | — | 88.9% | — | 88% | — |
| TriviaQA | Keyword filter + FLAN-T5 (rerank off) | 68.6 | +2.2 | 73.6 | +1.7 | 91.72 | +0.52 | 88.7% | +0.1 | 90% | -1 |
| TriviaQA | Keyword filter only (rerank on) _(not run)_ | — | — | — | — | — | — | — | — | — | — |
| TriviaQA | Keyword filter only (rerank off) _(not run)_ | — | — | — | — | — | — | — | — | — | — |
| TriviaQA | FLAN-T5 summarizer only (rerank on) _(not run)_ | — | — | — | — | — | — | — | — | — | — |
| TriviaQA | FLAN-T5 summarizer only (rerank off) _(not run)_ | — | — | — | — | — | — | — | — | — | — |
| NQ-Open | **Ours: Keyword filter + FLAN-T5** (rerank on, reference) | 29.4 | — | 37.5 | — | 87.84 | — | 83.3% | — | 85% | — |
| NQ-Open | Keyword filter + FLAN-T5 (rerank off) | 27.4 | +2.0 | 38.0 | -0.5 | 88.14 | -0.30 | 82.8% | +0.5 | 83% | +3 |
| NQ-Open | Keyword filter only (rerank on) _(not run)_ | — | — | — | — | — | — | — | — | — | — |
| NQ-Open | Keyword filter only (rerank off) _(not run)_ | — | — | — | — | — | — | — | — | — | — |
| NQ-Open | FLAN-T5 summarizer only (rerank on) _(not run)_ | — | — | — | — | — | — | — | — | — | — |
| NQ-Open | FLAN-T5 summarizer only (rerank off) _(not run)_ | — | — | — | — | — | — | — | — | — | — |
| HotpotQA | **Ours: Keyword filter + FLAN-T5** (rerank on, reference) | 23.2 | — | 31.8 | — | 89.18 | — | 88.4% | — | 75% | — |
| HotpotQA | Keyword filter + FLAN-T5 (rerank off) | 22.6 | +0.6 | 31.0 | +0.8 | 88.94 | +0.24 | 88.3% | +0.1 | 75% | +0 |
| HotpotQA | Keyword filter only (rerank on) _(not run)_ | — | — | — | — | — | — | — | — | — | — |
| HotpotQA | Keyword filter only (rerank off) _(not run)_ | — | — | — | — | — | — | — | — | — | — |
| HotpotQA | FLAN-T5 summarizer only (rerank on) _(not run)_ | — | — | — | — | — | — | — | — | — | — |
| HotpotQA | FLAN-T5 summarizer only (rerank off) _(not run)_ | — | — | — | — | — | — | — | — | — | — |
| 2WikiMultihopQA | **Ours: Keyword filter + FLAN-T5** (rerank on, reference) | 28.0 | — | 32.4 | — | 88.45 | — | 89.8% | — | 97% | — |
| 2WikiMultihopQA | Keyword filter + FLAN-T5 (rerank off) | 25.8 | +2.2 | 30.6 | +1.8 | 88.42 | +0.02 | 89.8% | +0.0 | 87% | +10 |
| 2WikiMultihopQA | Keyword filter only (rerank on) _(not run)_ | — | — | — | — | — | — | — | — | — | — |
| 2WikiMultihopQA | Keyword filter only (rerank off) _(not run)_ | — | — | — | — | — | — | — | — | — | — |
| 2WikiMultihopQA | FLAN-T5 summarizer only (rerank on) _(not run)_ | — | — | — | — | — | — | — | — | — | — |
| 2WikiMultihopQA | FLAN-T5 summarizer only (rerank off) _(not run)_ | — | — | — | — | — | — | — | — | — | — |
| MuSiQue | **Ours: Keyword filter + FLAN-T5** (rerank on, reference) | 7.8 | — | 16.7 | — | 85.71 | — | 86.7% | — | 91% | — |
| MuSiQue | Keyword filter + FLAN-T5 (rerank off) | 5.6 | +2.2 | 13.6 | +3.1 | 85.53 | +0.17 | 86.6% | +0.1 | 93% | -3 |
| MuSiQue | Keyword filter only (rerank on) _(not run)_ | — | — | — | — | — | — | — | — | — | — |
| MuSiQue | Keyword filter only (rerank off) _(not run)_ | — | — | — | — | — | — | — | — | — | — |
| MuSiQue | FLAN-T5 summarizer only (rerank on) _(not run)_ | — | — | — | — | — | — | — | — | — | — |
| MuSiQue | FLAN-T5 summarizer only (rerank off) _(not run)_ | — | — | — | — | — | — | — | — | — | — |

> **Gap:** no cells in this grid for FLAN-T5 summarizer only or Keyword filter only. Run them with `scripts/run_grid.py --conditions keyword_only summarizer_only` and regenerate; the rows above fill in automatically.

## Table 3 — Ours vs in-pipeline controls

Paired on identical questions. The uncompressed cell is the accuracy ceiling; truncation is the matched-budget control that spends the same tokens without query-aware selection.

**Retrieval: rerank off**

| Dataset | System | EM | ΔEM | F1 | ΔF1 | BERTScore | ΔBERT | Token redn | ΔRedn | Retention | ΔRet |
|---|---|---|---|---|---|---|---|---|---|---|---|
| TriviaQA | **Ours: Keyword filter + FLAN-T5** | 68.6 | — | 73.6 | — | 91.72 | — | 88.7% | — | 90% | — |
| TriviaQA | Uncompressed | 76.6 | -8.0 | 82.2 | -8.7 | 92.49 | -0.77 | 0.0% | +88.7 | 100% | -10 |
| TriviaQA | Truncate (matched budget) | 63.8 | +4.8 | 68.9 | +4.6 | 91.24 | +0.47 | 90.1% | -1.3 | 83% | +6 |
| NQ-Open | **Ours: Keyword filter + FLAN-T5** | 27.4 | — | 38.0 | — | 88.14 | — | 82.8% | — | 83% | — |
| NQ-Open | Uncompressed | 33.2 | -5.8 | 44.4 | -6.4 | 88.47 | -0.32 | 0.0% | +82.8 | 100% | -17 |
| NQ-Open | Truncate (matched budget) | 24.8 | +2.6 | 35.1 | +2.9 | 87.19 | +0.95 | 83.9% | -1.1 | 75% | +8 |
| HotpotQA | **Ours: Keyword filter + FLAN-T5** | 22.6 | — | 31.0 | — | 88.94 | — | 88.3% | — | 75% | — |
| HotpotQA | Uncompressed | 30.2 | -7.6 | 40.1 | -9.1 | 89.78 | -0.84 | 0.0% | +88.3 | 100% | -25 |
| HotpotQA | Truncate (matched budget) | 17.8 | +4.8 | 25.2 | +5.8 | 87.81 | +1.13 | 89.6% | -1.4 | 59% | +16 |
| 2WikiMultihopQA | **Ours: Keyword filter + FLAN-T5** | 25.8 | — | 30.6 | — | 88.42 | — | 89.8% | — | 87% | — |
| 2WikiMultihopQA | Uncompressed | 29.8 | -4.0 | 36.4 | -5.8 | 89.20 | -0.78 | 0.0% | +89.8 | 100% | -13 |
| 2WikiMultihopQA | Truncate (matched budget) | 28.0 | -2.2 | 31.5 | -1.0 | 88.68 | -0.26 | 91.4% | -1.6 | 94% | -7 |
| MuSiQue | **Ours: Keyword filter + FLAN-T5** | 5.6 | — | 13.6 | — | 85.53 | — | 86.6% | — | 93% | — |
| MuSiQue | Uncompressed | 6.0 | -0.4 | 14.0 | -0.5 | 85.23 | +0.30 | 0.0% | +86.6 | 100% | -7 |
| MuSiQue | Truncate (matched budget) | 3.8 | +1.8 | 10.5 | +3.0 | 85.01 | +0.52 | 88.2% | -1.6 | 63% | +30 |

**Retrieval: rerank on**

| Dataset | System | EM | ΔEM | F1 | ΔF1 | BERTScore | ΔBERT | Token redn | ΔRedn | Retention | ΔRet |
|---|---|---|---|---|---|---|---|---|---|---|---|
| TriviaQA | **Ours: Keyword filter + FLAN-T5** | 70.8 | — | 75.3 | — | 92.23 | — | 88.9% | — | 88% | — |
| TriviaQA | Uncompressed | 80.2 | -9.4 | 84.9 | -9.6 | 92.98 | -0.75 | 0.0% | +88.9 | 100% | -12 |
| TriviaQA | Truncate (matched budget) | 63.8 | +7.0 | 69.0 | +6.2 | 91.09 | +1.14 | 90.0% | -1.2 | 80% | +9 |
| NQ-Open | **Ours: Keyword filter + FLAN-T5** | 29.4 | — | 37.5 | — | 87.84 | — | 83.3% | — | 85% | — |
| NQ-Open | Uncompressed | 34.4 | -5.0 | 46.1 | -8.6 | 88.53 | -0.69 | 0.0% | +83.3 | 100% | -15 |
| NQ-Open | Truncate (matched budget) | 24.6 | +4.8 | 34.3 | +3.2 | 87.21 | +0.63 | 84.0% | -0.8 | 72% | +14 |
| HotpotQA | **Ours: Keyword filter + FLAN-T5** | 23.2 | — | 31.8 | — | 89.18 | — | 88.4% | — | 75% | — |
| HotpotQA | Uncompressed | 30.8 | -7.6 | 41.1 | -9.3 | 89.97 | -0.79 | 0.0% | +88.4 | 100% | -25 |
| HotpotQA | Truncate (matched budget) | 19.8 | +3.4 | 28.3 | +3.5 | 88.19 | +0.99 | 89.6% | -1.2 | 64% | +11 |
| 2WikiMultihopQA | **Ours: Keyword filter + FLAN-T5** | 28.0 | — | 32.4 | — | 88.45 | — | 89.8% | — | 97% | — |
| 2WikiMultihopQA | Uncompressed | 29.0 | -1.0 | 35.2 | -2.8 | 89.03 | -0.59 | 0.0% | +89.8 | 100% | -3 |
| 2WikiMultihopQA | Truncate (matched budget) | 26.0 | +2.0 | 30.2 | +2.2 | 88.61 | -0.16 | 91.4% | -1.5 | 90% | +7 |
| MuSiQue | **Ours: Keyword filter + FLAN-T5** | 7.8 | — | 16.7 | — | 85.71 | — | 86.7% | — | 91% | — |
| MuSiQue | Uncompressed | 8.6 | -0.8 | 17.1 | -0.4 | 85.54 | +0.17 | 0.0% | +86.7 | 100% | -9 |
| MuSiQue | Truncate (matched budget) | 3.8 | +4.0 | 11.6 | +5.1 | 85.13 | +0.57 | 88.1% | -1.4 | 44% | +47 |

## Token reduction and accuracy retention

**Retrieval: rerank off**

| Dataset | Condition | Prompt tok | Context tok | Reduction | Ratio | EM | Retention | F1 |
|---|---|---|---|---|---|---|---|---|
| TriviaQA | Uncompressed | 4777 | 4657 | 0.0% | 1.0× | 76.6 | 100% | 82.2 |
| TriviaQA | Truncate (matched budget) | 583 | 463 | 90.1% | 10.1× | 63.8 | 83% | 68.9 |
| TriviaQA | Keyword filter + FLAN-T5 | 645 | 525 | 88.7% | 8.9× | 68.6 | 90% | 73.6 |
| NQ-Open | Uncompressed | 4681 | 4570 | 0.0% | 1.0× | 33.2 | 100% | 44.4 |
| NQ-Open | Truncate (matched budget) | 846 | 734 | 83.9% | 6.2× | 24.8 | 75% | 35.1 |
| NQ-Open | Keyword filter + FLAN-T5 | 898 | 787 | 82.8% | 5.8× | 27.4 | 83% | 38.0 |
| HotpotQA | Uncompressed | 4888 | 4742 | 0.0% | 1.0× | 30.2 | 100% | 40.1 |
| HotpotQA | Truncate (matched budget) | 637 | 491 | 89.6% | 9.7× | 17.8 | 59% | 25.2 |
| HotpotQA | Keyword filter + FLAN-T5 | 702 | 556 | 88.3% | 8.5× | 22.6 | 75% | 31.0 |
| 2WikiMultihopQA | Uncompressed | 5130 | 4974 | 0.0% | 1.0× | 29.8 | 100% | 36.4 |
| 2WikiMultihopQA | Truncate (matched budget) | 582 | 426 | 91.4% | 11.7× | 28.0 | 94% | 31.5 |
| 2WikiMultihopQA | Keyword filter + FLAN-T5 | 661 | 505 | 89.8% | 9.8× | 25.8 | 87% | 30.6 |
| MuSiQue | Uncompressed | 4830 | 4661 | 0.0% | 1.0× | 6.0 | 100% | 14.0 |
| MuSiQue | Truncate (matched budget) | 721 | 552 | 88.2% | 8.4× | 3.8 | 63% | 10.5 |
| MuSiQue | Keyword filter + FLAN-T5 | 793 | 625 | 86.6% | 7.5× | 5.6 | 93% | 13.6 |

**Retrieval: rerank on**

| Dataset | Condition | Prompt tok | Context tok | Reduction | Ratio | EM | Retention | F1 |
|---|---|---|---|---|---|---|---|---|
| TriviaQA | Uncompressed | 4768 | 4648 | 0.0% | 1.0× | 80.2 | 100% | 84.9 |
| TriviaQA | Truncate (matched budget) | 584 | 463 | 90.0% | 10.0× | 63.8 | 80% | 69.0 |
| TriviaQA | Keyword filter + FLAN-T5 | 638 | 518 | 88.9% | 9.0× | 70.8 | 88% | 75.3 |
| NQ-Open | Uncompressed | 4712 | 4601 | 0.0% | 1.0× | 34.4 | 100% | 46.1 |
| NQ-Open | Truncate (matched budget) | 845 | 734 | 84.0% | 6.3× | 24.6 | 72% | 34.3 |
| NQ-Open | Keyword filter + FLAN-T5 | 880 | 769 | 83.3% | 6.0× | 29.4 | 85% | 37.5 |
| HotpotQA | Uncompressed | 4870 | 4725 | 0.0% | 1.0× | 30.8 | 100% | 41.1 |
| HotpotQA | Truncate (matched budget) | 638 | 492 | 89.6% | 9.6× | 19.8 | 64% | 28.3 |
| HotpotQA | Keyword filter + FLAN-T5 | 694 | 548 | 88.4% | 8.6× | 23.2 | 75% | 31.8 |
| 2WikiMultihopQA | Uncompressed | 5087 | 4931 | 0.0% | 1.0× | 29.0 | 100% | 35.2 |
| 2WikiMultihopQA | Truncate (matched budget) | 582 | 426 | 91.4% | 11.6× | 26.0 | 90% | 30.2 |
| 2WikiMultihopQA | Keyword filter + FLAN-T5 | 659 | 503 | 89.8% | 9.8× | 28.0 | 97% | 32.4 |
| MuSiQue | Uncompressed | 4815 | 4647 | 0.0% | 1.0× | 8.6 | 100% | 17.1 |
| MuSiQue | Truncate (matched budget) | 722 | 553 | 88.1% | 8.4× | 3.8 | 44% | 11.6 |
| MuSiQue | Keyword filter + FLAN-T5 | 785 | 616 | 86.7% | 7.5× | 7.8 | 91% | 16.7 |

## Sample counts

Every cell evaluates the same question set, so all comparisons are paired.

| Dataset | n per cell | Conditions | Rerank settings | Cells | Total answers |
|---|---|---|---|---|---|
| TriviaQA | 500 | 3 | 2 | 6 | 3000 |
| NQ-Open | 500 | 3 | 2 | 6 | 3000 |
| HotpotQA | 500 | 3 | 2 | 6 | 3000 |
| 2WikiMultihopQA | 500 | 3 | 2 | 6 | 3000 |
| MuSiQue | 500 | 3 | 2 | 6 | 3000 |
| **Total** | | | | **30** | **15000** |

## Token F1 — all conditions

F1 degrades more gracefully than EM and stays informative where EM floors.

**Retrieval: rerank off**

| Dataset | Uncompressed | Truncate (matched budget) | Keyword filter + FLAN-T5 | Δ main vs uncompressed | Retention |
|---|---|---|---|---|---|
| TriviaQA | 82.2 [79.1, 85.2] | 68.9 [65.0, 72.8] | 73.6 [69.8, 77.2] | -8.7 | 89% |
| NQ-Open | 44.4 [40.5, 48.4] | 35.1 [31.3, 38.9] | 38.0 [34.2, 41.8] | -6.4 | 86% |
| HotpotQA | 40.1 [36.2, 44.1] | 25.2 [21.8, 28.7] | 31.0 [27.3, 34.8] | -9.1 | 77% |
| 2WikiMultihopQA | 36.4 [32.5, 40.3] | 31.5 [27.6, 35.5] | 30.6 [26.8, 34.5] | -5.8 | 84% |
| MuSiQue | 14.0 [11.6, 16.6] | 10.5 [8.4, 12.7] | 13.6 [11.1, 16.1] | -0.5 | 97% |

**Retrieval: rerank on**

| Dataset | Uncompressed | Truncate (matched budget) | Keyword filter + FLAN-T5 | Δ main vs uncompressed | Retention |
|---|---|---|---|---|---|
| TriviaQA | 84.9 [81.9, 87.8] | 69.0 [65.1, 72.9] | 75.3 [71.6, 78.8] | -9.6 | 89% |
| NQ-Open | 46.1 [42.1, 50.0] | 34.3 [30.5, 38.0] | 37.5 [33.6, 41.4] | -8.6 | 81% |
| HotpotQA | 41.1 [37.1, 45.1] | 28.3 [24.7, 31.9] | 31.8 [28.1, 35.6] | -9.3 | 77% |
| 2WikiMultihopQA | 35.2 [31.3, 39.1] | 30.2 [26.4, 34.0] | 32.4 [28.6, 36.2] | -2.8 | 92% |
| MuSiQue | 17.1 [14.4, 20.0] | 11.6 [9.4, 13.9] | 16.7 [14.0, 19.5] | -0.4 | 97% |

Values are mean with 95% bootstrap CI.

## BERTScore F1 — all conditions

Semantic similarity to the reference answer. Because it does not require surface-form agreement, it separates *phrasing* loss from *content* loss - a compressor that paraphrases is penalised by EM but not by BERTScore.

**Retrieval: rerank off**

| Dataset | Uncompressed | Truncate (matched budget) | Keyword filter + FLAN-T5 | Δ main vs uncompressed |
|---|---|---|---|---|
| TriviaQA | 92.49 | 91.24 | 91.72 | -0.77 |
| NQ-Open | 88.47 | 87.19 | 88.14 | -0.32 |
| HotpotQA | 89.78 | 87.81 | 88.94 | -0.84 |
| 2WikiMultihopQA | 89.20 | 88.68 | 88.42 | -0.78 |
| MuSiQue | 85.23 | 85.01 | 85.53 | +0.30 |

**Retrieval: rerank on**

| Dataset | Uncompressed | Truncate (matched budget) | Keyword filter + FLAN-T5 | Δ main vs uncompressed |
|---|---|---|---|---|
| TriviaQA | 92.98 | 91.09 | 92.23 | -0.75 |
| NQ-Open | 88.53 | 87.21 | 87.84 | -0.69 |
| HotpotQA | 89.97 | 88.19 | 89.18 | -0.79 |
| 2WikiMultihopQA | 89.03 | 88.61 | 88.45 | -0.59 |
| MuSiQue | 85.54 | 85.13 | 85.71 | +0.17 |

## Variants vs the proposed method (Keyword filter + FLAN-T5)

Paired comparison on identical questions. **ΔEM** and **ΔF1** are *other minus main*, so negative means the proposed method wins. `p` is McNemar's test on exact-match correctness.

| Dataset | rerank | Variant | ΔEM | ΔF1 | p (EM) | Δ context tok |
|---|---|---|---|---|---|---|
| TriviaQA | off | Uncompressed | +8.0 | +8.7 | **0.0001** | +4132 |
| TriviaQA | off | Truncate (matched budget) | -4.8 | -4.6 | **0.0255** | -62 |
| TriviaQA | on | Uncompressed | +9.4 | +9.6 | **0.0000** | +4130 |
| TriviaQA | on | Truncate (matched budget) | -7.0 | -6.2 | **0.0005** | -54 |
| NQ-Open | off | Uncompressed | +5.8 | +6.4 | **0.0033** | +3784 |
| NQ-Open | off | Truncate (matched budget) | -2.6 | -2.9 | 0.2134 | -52 |
| NQ-Open | on | Uncompressed | +5.0 | +8.6 | **0.0119** | +3832 |
| NQ-Open | on | Truncate (matched budget) | -4.8 | -3.2 | **0.0121** | -35 |
| HotpotQA | off | Uncompressed | +7.6 | +9.1 | **0.0002** | +4186 |
| HotpotQA | off | Truncate (matched budget) | -4.8 | -5.8 | **0.0075** | -65 |
| HotpotQA | on | Uncompressed | +7.6 | +9.3 | **0.0001** | +4177 |
| HotpotQA | on | Truncate (matched budget) | -3.4 | -3.5 | 0.0611 | -56 |
| 2WikiMultihopQA | off | Uncompressed | +4.0 | +5.8 | 0.0650 | +4469 |
| 2WikiMultihopQA | off | Truncate (matched budget) | +2.2 | +1.0 | 0.2891 | -80 |
| 2WikiMultihopQA | on | Uncompressed | +1.0 | +2.8 | 0.7067 | +4429 |
| 2WikiMultihopQA | on | Truncate (matched budget) | -2.0 | -2.2 | 0.3775 | -76 |
| MuSiQue | off | Uncompressed | +0.4 | +0.5 | 0.8501 | +4037 |
| MuSiQue | off | Truncate (matched budget) | -1.8 | -3.0 | 0.1237 | -73 |
| MuSiQue | on | Uncompressed | +0.8 | +0.4 | 0.6265 | +4030 |
| MuSiQue | on | Truncate (matched budget) | -4.0 | -5.1 | **0.0005** | -63 |

Bold p-values are significant at 0.05. A negative Δ with a positive Δ context means the variant is worse *and* not cheaper.

## Total token spend across all datasets

| Condition | rerank | Total prompt tokens | Saved vs uncompressed |
|---|---|---|---|
| Uncompressed | off | 12.15M | — |
| Truncate (matched budget) | off | 1.68M | 86.1% |
| Keyword filter + FLAN-T5 | off | 1.85M | 84.8% |
| Uncompressed | on | 12.13M | — |
| Truncate (matched budget) | on | 1.69M | 86.1% |
| Keyword filter + FLAN-T5 | on | 1.83M | 84.9% |

## Efficiency — exact matches per 1k prompt tokens

| Dataset | rerank | Uncompressed | Truncate (matched budget) | Keyword filter + FLAN-T5 |
|---|---|---|---|---|
| TriviaQA | off | 16.03 | 109.47 | 106.32 |
| TriviaQA | on | 16.82 | 109.33 | 111.02 |
| NQ-Open | off | 7.09 | 29.33 | 30.52 |
| NQ-Open | on | 7.30 | 29.11 | 33.42 |
| HotpotQA | off | 6.18 | 27.95 | 32.19 |
| HotpotQA | on | 6.32 | 31.05 | 33.45 |
| 2WikiMultihopQA | off | 5.81 | 48.14 | 39.02 |
| 2WikiMultihopQA | on | 5.70 | 44.67 | 42.52 |
| MuSiQue | off | 1.24 | 5.27 | 7.06 |
| MuSiQue | on | 1.79 | 5.27 | 9.94 |

## Compression ratio vs published compressors

Published ratios are from CompAct (EMNLP 2024) Table 2 on HotpotQA with a LLaMA3-8B reader. They are context for where a training-free compressor sits, not a like-for-like comparison.

| System | Compression | Trained? |
|---|---|---|
| **Ours: keyword + FLAN-T5 (rerank off)** | **8.5×** | no |
| **Ours: keyword + FLAN-T5 (rerank on)** | **8.6×** | no |
| CompAct | 47.6× | yes |
| AutoCompressors | 35.4× | yes |
| RECOMP (extractive) | 34.3× | yes |
| LongLLMLingua | 3.4× | yes |

## Supporting-fact retention (multi-hop)

Share of gold supporting sentences surviving compression. This is the mechanism behind multi-hop accuracy loss and is measured over more items than EM, so it is the more stable signal.

| Dataset | rerank | Uncompressed | Truncate (matched budget) | Keyword filter + FLAN-T5 |
|---|---|---|---|---|
| HotpotQA | off | 93% | 2% | 13% |
| HotpotQA | on | 95% | 3% | 13% |
| 2WikiMultihopQA | off | 85% | 0% | 4% |
| 2WikiMultihopQA | on | 88% | 0% | 4% |
| MuSiQue | off | 74% | 0% | 3% |
| MuSiQue | on | 73% | 0% | 3% |

## Latency (ms/question, end to end)

| Dataset | rerank | Uncompressed | Truncate (matched budget) | Keyword filter + FLAN-T5 |
|---|---|---|---|---|
| TriviaQA | off | 4670 | 2479 | 4658 |
| TriviaQA | on | 4877 | 2669 | 4790 |
| NQ-Open | off | 4628 | 2490 | 7357 |
| NQ-Open | on | 4899 | 2721 | 7424 |
| HotpotQA | off | 4712 | 2452 | 4846 |
| HotpotQA | on | 4923 | 2722 | 5115 |
| 2WikiMultihopQA | off | 5023 | 2492 | 4240 |
| 2WikiMultihopQA | on | 5235 | 2755 | 4766 |
| MuSiQue | off | 4757 | 2352 | 5388 |
| MuSiQue | on | 4986 | 2621 | 5457 |
