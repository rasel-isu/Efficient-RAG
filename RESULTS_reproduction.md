# Tier-1 Reproduction — Detailed Analysis

Uncompressed ("Raw Document") retrieval-augmented QA on five benchmarks, run to validate the pipeline against published baselines before any compression experiments.

## Configuration

| | |
|---|---|
| Reader | `meta-llama/Llama-3.2-3B-Instruct` |
| Retriever | Contriever-MSMARCO (`contriever`), dot-product, no reranking |
| Corpus | DPR Wikipedia 2018, 21,015,324 passages |
| Depth | top-30 |
| Prompt | 5-shot in-context (RECOMP format), greedy decoding |
| Split / size | dev, n=500 per dataset |

## Token cost — the baseline this project aims to reduce

This project's contribution is token efficiency: cutting the tokens sent to the generator while keeping accuracy acceptable. These are the **uncompressed** costs that the compression conditions are measured against.

| Dataset | prompt tokens | of which context | compressible share |
|---|---|---|---|
| TriviaQA | **4777** | 4657 | 97.5% |
| NQ-Open | **4681** | 4570 | 97.6% |
| HotpotQA | **4888** | 4742 | 97.0% |
| 2WikiMultihopQA | **5130** | 4974 | 97.0% |
| MuSiQue | **4830** | 4661 | 96.5% |

The remainder is the 5-shot in-context block (~100-145 tokens) plus the question. That part is a fixed cost and cannot be compressed away, but at top-30 it is under 3% of the prompt, so essentially the whole prompt is addressable.

Across all 2500 questions in this run the uncompressed setting consumed **12.15M prompt tokens**.

## Headline accuracy

| Dataset | EM | 95% CI | F1 | 95% CI | ctx tokens |
|---|---|---|---|---|---|
| TriviaQA | **76.6** | [72.8, 80.2] | **82.2** | [79.1, 85.2] | 4657 |
| NQ-Open | **33.2** | [29.2, 37.4] | **44.4** | [40.5, 48.4] | 4570 |
| HotpotQA | **30.2** | [26.2, 34.4] | **40.1** | [36.2, 44.1] | 4742 |
| 2WikiMultihopQA | **29.8** | [25.8, 33.8] | **36.4** | [32.5, 40.3] | 4974 |
| MuSiQue | **6.0** | [4.0, 8.2] | **14.0** | [11.6, 16.6] | 4661 |

## Comparison with published methods

CompAct (EMNLP 2024) Table 2 — reader **LLaMA3-8B**, same retriever/corpus/depth. Our reader is 3B, so absolute parity is not expected; the comparison shows where this pipeline sits relative to published systems.

**TriviaQA**

| System | Reader | EM | F1 | ΔEM vs ours |
|---|---|---|---|---|
| **Ours (uncompressed)** | Llama-3.2-3B | **76.6** | **82.2** | — |
| Raw Document | LLaMA3-8B | 68.9 | 77.1 | +7.7 |
| RECOMP (extractive) | LLaMA3-8B | 67.6 | 74.1 | +9.0 |
| LongLLMLingua | LLaMA3-8B | 64.0 | 70.8 | +12.6 |
| AutoCompressors | LLaMA3-8B | 55.3 | 64.3 | +21.3 |
| CompAct | LLaMA3-8B | 65.4 | 74.9 | +11.2 |

**NQ-Open**

| System | Reader | EM | F1 | ΔEM vs ours |
|---|---|---|---|---|
| **Ours (uncompressed)** | Llama-3.2-3B | **33.2** | **44.4** | — |
| Raw Document | LLaMA3-8B | 39.0 | 51.3 | -5.8 |
| RECOMP (extractive) | LLaMA3-8B | 34.6 | 45.1 | -1.4 |
| LongLLMLingua | LLaMA3-8B | 27.7 | 40.6 | +5.5 |
| AutoCompressors | LLaMA3-8B | 17.3 | 31.8 | +15.9 |
| CompAct | LLaMA3-8B | 38.4 | 50.0 | -5.2 |

**HotpotQA**

| System | Reader | EM | F1 | ΔEM vs ours |
|---|---|---|---|---|
| **Ours (uncompressed)** | Llama-3.2-3B | **30.2** | **40.1** | — |
| Raw Document | LLaMA3-8B | 29.4 | 40.3 | +0.8 |
| RECOMP (extractive) | LLaMA3-8B | 29.7 | 39.9 | +0.5 |
| LongLLMLingua | LLaMA3-8B | 25.6 | 35.3 | +4.6 |
| AutoCompressors | LLaMA3-8B | 18.4 | 28.4 | +11.8 |
| CompAct | LLaMA3-8B | 35.5 | 46.9 | -5.3 |
| Oracle | LLaMA3-8B | 39.9 | 51.2 | -9.7 |

**2WikiMultihopQA**

| System | Reader | EM | F1 | ΔEM vs ours |
|---|---|---|---|---|
| **Ours (uncompressed)** | Llama-3.2-3B | **29.8** | **36.4** | — |
| Raw Document | LLaMA3-8B | 25.4 | 31.2 | +4.4 |
| RECOMP (extractive) | LLaMA3-8B | 29.9 | 34.9 | -0.1 |
| LongLLMLingua | LLaMA3-8B | 27.9 | 32.9 | +1.9 |
| AutoCompressors | LLaMA3-8B | 19.0 | 24.5 | +10.8 |
| CompAct | LLaMA3-8B | 31.0 | 37.1 | -1.2 |
| Oracle | LLaMA3-8B | 37.4 | 43.2 | -7.6 |

**MuSiQue**

| System | Reader | EM | F1 | ΔEM vs ours |
|---|---|---|---|---|
| **Ours (uncompressed)** | Llama-3.2-3B | **6.0** | **14.0** | — |
| Raw Document | LLaMA3-8B | 6.5 | 15.6 | -0.5 |
| RECOMP (extractive) | LLaMA3-8B | 6.7 | 15.7 | -0.7 |
| LongLLMLingua | LLaMA3-8B | 4.8 | 13.5 | +1.2 |
| AutoCompressors | LLaMA3-8B | 3.9 | 11.9 | +2.1 |
| CompAct | LLaMA3-8B | 8.7 | 18.1 | -2.7 |
| Oracle | LLaMA3-8B | 14.2 | 23.6 | -8.2 |

### RECOMP (ICLR 2024) — Flan-UL2 20B, Contriever top-5

A different reader and retrieval depth, so this is context rather than a like-for-like comparison. The `T5 (off-the-shelf)` row is the closest published analogue to this project's original method.

**TriviaQA** (ours uncompressed: EM 76.6 / F1 82.2)

| System | EM | F1 |
|---|---|---|
| Top 5 documents | 62.4 | 70.1 |
| Top 1 document | 57.8 | 64.9 |
| no retrieval | 49.3 | 54.9 |
| T5 (off-the-shelf) | 55.2 | 62.3 |
| RECOMP abstractive | 58.7 | 66.3 |
| RECOMP extractive | 59.0 | 65.3 |

**NQ-Open** (ours uncompressed: EM 33.2 / F1 44.4)

| System | EM | F1 |
|---|---|---|
| Top 5 documents | 39.4 | 48.3 |
| Top 1 document | 33.1 | 41.5 |
| no retrieval | 22.0 | 29.4 |
| T5 (off-the-shelf) | 25.9 | 34.6 |
| RECOMP abstractive | 37.0 | 45.5 |
| RECOMP extractive | 36.6 | 44.2 |

**HotpotQA** (ours uncompressed: EM 30.2 / F1 40.1)

| System | EM | F1 |
|---|---|---|
| Top 5 documents | 32.8 | 43.9 |
| Top 1 document | 28.8 | 40.6 |
| no retrieval | 17.8 | 26.1 |
| T5 (off-the-shelf) | 23.2 | 33.2 |
| RECOMP abstractive | 28.2 | 37.9 |
| RECOMP extractive | 30.4 | 40.1 |

## Retrieval diagnostics — evidence recall

Retrieval is over the full 21M-passage Wikipedia index, so no retrieved passage carries a dataset `is_gold` flag. Instead we measure whether the gold supporting sentences actually appear in the retrieved top-30 (token recall >= 0.6 per sentence), and split accuracy by whether the evidence was found. Only the multi-hop sets ship sentence-level supporting facts.

| Dataset | evidence recall | EM when evidence found | EM when missed | n (found/missed) |
|---|---|---|---|---|
| TriviaQA | n/a (no supporting-fact annotations) | — | — | — |
| NQ-Open | n/a (no supporting-fact annotations) | — | — | — |
| HotpotQA | 92.4% | 30.4 | 14.3 | 493/7 |
| 2WikiMultihopQA | 82.3% | 29.7 | 31.0 | 471/29 |
| MuSiQue | 73.2% | 6.9 | 0.0 | 433/67 |

## Cost and latency (per question)

Retrieval is negligible (~30-40 ms over 21M passages); generation dominates, and generation time is driven by encoding the retrieved context. Fewer context tokens therefore buys latency as well as cost.

| Dataset | prompt tok | completion tok | retrieval ms | generation ms | total ms |
|---|---|---|---|---|---|
| TriviaQA | 4777 | 94.2 | 30 | 4640 | 4670 |
| NQ-Open | 4681 | 95.2 | 30 | 4597 | 4628 |
| HotpotQA | 4888 | 91.5 | 32 | 4681 | 4712 |
| 2WikiMultihopQA | 5130 | 94.4 | 31 | 4991 | 5023 |
| MuSiQue | 4830 | 95.9 | 31 | 4726 | 4757 |

## Accuracy by question type

**TriviaQA** — Other: 76.9% (n=251), What: 69.3% (n=101), Which: 83.7% (n=86), Who: 79.6% (n=49), How: 66.7% (n=6), Where: 50.0% (n=4), When: 100.0% (n=3)

**NQ-Open** — Who: 46.2% (n=156), When: 32.7% (n=101), Other: 23.0% (n=74), What: 37.5% (n=72), Where: 10.2% (n=59), How: 19.2% (n=26), Which: 62.5% (n=8), Why: 0.0% (n=2), Yes/No: 50.0% (n=2)

**HotpotQA** — bridge: 27.7% (n=422), comparison: 43.6% (n=78)

**2WikiMultihopQA** — compositional: 7.9% (n=202), bridge_comparison: 52.3% (n=130), comparison: 58.7% (n=104), inference: 6.2% (n=64)

**MuSiQue** — 2-hop: 10.3% (n=252), 3-hop: 1.2% (n=171), 4-hop: 2.6% (n=77)
