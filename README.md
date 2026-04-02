# Efficient Token Optimization in RAG Pipelines for HPC

> A query-aware summarization approach that reduces token usage by **60%** while retaining **95% of baseline accuracy** in Retrieval-Augmented Generation pipelines.

[![Python](https://img.shields.io/badge/Python-3.8+-blue)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![HPC](https://img.shields.io/badge/Use%20Case-HPC%20Pipelines-orange)](http://138.2.71.205:3000/)


---

## Table of Contents

- [Overview](#overview)
- [Motivation](#motivation)
- [Proposed Method](#proposed-method)
- [Installation](#installation)
- [Usage](#usage)
- [Experiment Results](#experiment-results)
- [Cost-Benefit Analysis](#cost-benefit-analysis)
- [Performance by Question Type](#performance-by-question-type)
- [Advanced Metrics](#advanced-metrics)
- [Recommendations](#recommendations)
- [Project Structure](#project-structure)
- [Future Work](#future-work)

---

## Overview

Standard RAG pipelines retrieve documents and pass them **in full** to the LLM for answer generation. Most of that retrieved content is irrelevant to the query — yet it still contributes to the total token count, inflating API costs and latency.

This project introduces a **two-stage context compression** step inserted between retrieval and generation:

1. **Keyword-based sentence filtering** — select the top-k sentences most relevant to the query via word overlap
2. **LM-based query-aware summarization** — compress those sentences using a local T5 model (Small / Base / Large)

The final prompt sent to the LLM is: `query + compressed summary` instead of `query + full retrieved documents`.

The approach is **model-agnostic** and designed for **HPC batch pipelines** with high query volumes.

---

## Motivation

| Problem | Impact |
|---------|--------|
| Token bloat | Baseline RAG uses 604K tokens for 918 questions |
| API cost at scale | $0.30 per 918 questions; compounds to thousands at HPC scale |
| No context filtering | Same context sent regardless of query specificity |

---

## Proposed Method

### Pipeline

```
PDF Docs
   ↓
Text Cleaning (remove HTML)
   ↓
Chunking (SentenceSplitter, chunk_size=200, overlap=50)
   ↓
Embedding + Vector Index (Weaviate)
   ↓
Hybrid Search + Cross-Encoder Reranking (top-5)
   ↓
★ [NEW] Keyword-based Sentence Filtering
   ↓
★ [NEW] T5 Query-Aware Summarization
   ↓
GPT-3.5-Turbo Answer Generation
```

### Step Details

**Keyword-based Sentence Filtering**
- Extract sentences from retrieved documents
- Score by word overlap with the query
- Select top-k; fall back to first-k if no overlap exists

**LM Summarization**
- Prompt: `"Summarize for the question: {query}\n\nContext: {filtered_sentences}"`
- Runs locally — no additional API cost
- Three model sizes tested: T5-Small, T5-Base, T5-Large

**Final Prompt**
- `query + document_summary` (instead of full retrieved chunks)
- Sent to GPT-3.5-Turbo for answer generation

---

## Installation

> ⚠️ **Do not run on Windows** — Embedded Weaviate DB is not supported. See [issue #3315](https://github.com/weaviate/weaviate/issues/3315).

### Using Docker (recommended)

```bash
docker-compose up --build -d
# App available at http://0.0.0.0:3000/
```

### Local — Linux

```bash
pip3 install -r requirements.txt
```

### Local — Mac

```bash
python3 -m pip install --upgrade pip
pip3 install -r requirements_mac.txt
```

### Environment Setup

Create a `.env` file in the root directory:

```
OPENAI_API_KEY=your_openai_api_key_here
```

---

## Usage

### GUI

Visit http://0.0.0.0:3000/ after starting the app. [▶ Video walkthrough](https://youtu.be/lmzuVHEJJVM)

### Command Line

**Upload a PDF:**
```bash
python upload.py --pdf_file=your_document.pdf
```
[▶ Video](https://youtu.be/z_Xjxqk8E4g)

**Ask a question:**
```bash
python query.py --question="What is mt5?"
```
[▶ Video](https://youtu.be/H8mwEB64cJ0)

### Export Vector DB Data

```python
self.save_data_from_index_to_file(client)
# Output: index_data.json
```

---

## Experiment Results

**Dataset:** `rag-mini-wikipedia` · 918 QA pairs  
**Generator:** GPT-3.5-Turbo  
**Retrieval:** FAISS + Cross-Encoder reranking (top-5)

### Overall Performance

| Model | Exact Match | F1 Score | BERTScore | Tokens Used | Est. Cost |
|-------|-------------|----------|-----------|-------------|-----------|
| **Baseline RAG** | **58.28%** ★ | **0.704** ★ | **0.9474** ★ | 604,223 | $0.30 |
| T5-Small Summary ⭐ | 55.34% | 0.640 | 0.9389 | **240,580** (-60%) | **$0.12** |
| T5-Base Summary | 53.70% | 0.633 | 0.9397 | **231,525** (-62%) | **$0.12** |
| T5-Large Summary | 52.83% | 0.614 | 0.9360 | 241,491 (-60%) | $0.12 |

★ Best in category &nbsp;·&nbsp; ⭐ Recommended &nbsp;·&nbsp; (-%) Token reduction vs Baseline

**Key finding:** T5-Small achieves **95% of baseline accuracy at 40% of the cost**.

---

## Cost-Benefit Analysis

| Model | Tokens vs Baseline | Cost Savings | Accuracy Trade-off | Efficiency Score* |
|-------|-------------------|--------------|-------------------|-------------------|
| T5-Small | -60.2% (363K saved) | $0.18 saved | -2.9% | **4.61** ⭐ |
| T5-Base | -61.7% (373K saved) | $0.19 saved | -4.6% | 4.48 |
| T5-Large | -60.0% (363K saved) | $0.18 saved | -5.5% | 4.39 |
| Baseline | — | — | — | 1.94 |

*Efficiency Score = Accuracy / Cost (higher is better)*

### Decision Matrix

| Use Case | Recommended Model | Why |
|----------|------------------|-----|
| Accuracy-critical (research / benchmark) | Baseline RAG | Best exact match (58.28%), highest BERTScore |
| Cost-sensitive / high-volume | T5-Base Summary | Lowest token count (231K), cheapest at scale |
| Balanced production workload | **T5-Small Summary** ⭐ | Best efficiency ratio (4.61), 95% of accuracy |
| ❌ Avoid | T5-Large Summary | Lowest accuracy, no cost benefit over T5-Small |

> **T5 Model Size Paradox:** Larger T5 models do *not* improve summarization quality in this setting — T5-Small outperforms T5-Large. Bigger models likely over-compress and discard key facts.

---

## Performance by Question Type

| Question Type | Count | Baseline | T5-Small | T5-Base | T5-Large | Average |
|---------------|-------|----------|----------|---------|----------|---------|
| Yes/No | 420 | **89.8%** ✅ | 85.2% | 83.6% | 85.7% | 86.1% |
| When | 41 | 53.7% | 53.7% | 41.5% | 31.7% | 45.2% |
| Which | 11 | 45.5% | 36.4% | 45.5% | 36.4% | 40.9% |
| Other | 51 | 41.2% | 47.1% | 41.2% | 39.2% | 42.2% |
| Who | 54 | 40.7% | 29.6% | 31.5% | 29.6% | 32.9% |
| Where | 32 | 34.4% | 28.1% | 18.8% | 25.0% | 26.6% |
| What | 221 | 28.1% | 26.7% | 29.0% | 24.0% | 26.9% |
| How | 63 | 22.2% | 25.4% | 17.5% | 15.9% | 20.2% |
| Why | 25 | 4.0% 🔴 | 0.0% 🔴 | 4.0% 🔴 | 4.0% 🔴 | 3.0% |

**Insights:**
- ✅ All models excel at Yes/No questions (>83%) — binary retrieval works well
- 🔴 All models struggle with Why questions (<5%) — open-ended reasoning is a critical gap
- 📊 What questions make up 24% of the dataset but average only ~27% accuracy — high-impact improvement area
- 9× performance gap between the best (Yes/No) and worst (Why) question types

---

## Advanced Metrics

### BERTScore (Semantic Similarity)

| Model | Precision | Recall | F1 |
|-------|-----------|--------|----|
| Baseline | 0.9465 | 0.9490 | **0.9474** |
| T5-Small | 0.9405 | 0.9381 | 0.9389 |
| T5-Base | 0.9419 | 0.9383 | 0.9397 |
| T5-Large | 0.9393 | 0.9337 | 0.9360 |

All models score >0.93 BERTScore — answers are **semantically correct** even when they don't exactly match the ground truth string. The exact match gap between Baseline and T5 variants is partly a surface-level phrasing artifact, not a deep semantic failure.

### ROUGE Scores

| Model | ROUGE-1 | ROUGE-2 | ROUGE-L |
|-------|---------|---------|---------|
| Baseline | **0.724** | **0.166** | **0.719** |
| T5-Small | 0.653 | 0.127 | 0.651 |
| T5-Base | 0.649 | 0.133 | 0.646 |
| T5-Large | 0.626 | 0.117 | 0.623 |

### Semantic Similarity (Cosine)

| Model | Mean | Std Dev |
|-------|------|---------|
| Baseline | **0.796** | 0.288 |
| T5-Small | 0.748 | 0.329 |
| T5-Base | 0.749 | 0.328 |
| T5-Large | 0.712 | 0.356 |

### Error Summary

| Model | Correct / Total | Error Rate | Token Efficiency |
|-------|----------------|------------|-----------------|
| Baseline | 535 / 918 | 41.7% | 1.00× |
| T5-Small | 508 / 918 | 44.7% | **2.51×** |
| T5-Base | 493 / 918 | 46.3% | **2.61×** |
| T5-Large | 485 / 918 | 47.2% | 2.50× |

---

## Recommendations

**Use Baseline RAG when:**
- Accuracy is critical (58.28% vs 53–55% for T5)
- Cost is not a concern
- You need the highest semantic quality (0.947 BERTScore)

**Use T5-Small when (recommended default):**
- You need the best balance of performance and cost
- 55% accuracy is acceptable (only 5% drop)
- You want 60% token reduction at 40% of the cost
- Deploying at HPC scale with high query volume

**Use T5-Base when:**
- Minimizing tokens/cost is the top priority (62% reduction)
- Volume is very high and every cent matters

**Avoid T5-Large:**
- Lowest accuracy (52.83%) with no cost advantage
- Smaller T5 models outperform it — over-compression loses key facts

---

## Project Structure

```
.
├── upload.py                  # PDF ingestion script
├── query.py                   # CLI query interface
├── requirements.txt           # Linux dependencies
├── requirements_mac.txt       # Mac dependencies
├── docker-compose.yml         # Docker setup
├── .env                       # API keys (not committed)
├── index_data.json            # Exported vector DB data
└── src/
    ├── text_cleaner.py        # HTML/text cleaning
    ├── indexing.py            # Chunking + embedding
    ├── retriever.py           # Hybrid search + reranking
    ├── summarizer.py          # T5 query-aware summarization
    └── query_engine.py        # End-to-end pipeline
```

**Key components in code:**

```python
# Text cleaning
clean_text = TextCleaner(doc.text).clean()

# Chunking
Settings.text_splitter = SentenceSplitter(
    separator=" ", chunk_size=200, chunk_overlap=50,
    paragraph_separator="\n\n\n",
    secondary_chunking_regex="[^,.;。]+[,.;。]?",
    tokenizer=tiktoken.encoding_for_model(self.model_name).encode
)

# Embedding + indexing
index, nodes = indexing.get_index()

# Reranking
self.rerank = SentenceTransformerRerank(top_n=5, model=self.model_reranker)

# Hybrid retrieval
query_engine = self.index.as_query_engine(
    similarity_top_k=5,
    vector_store_query_mode="hybrid",
    alpha=0.5,
    node_postprocessors=[self.postproc, self.rerank],
)

# Answer generation
response = Retriever(index, nodes).get_response("What is t5?")
```

---

## Future Work

**High Priority**
- Chain-of-thought prompting for Why/How questions (currently <5% accuracy)
- Hybrid routing: use Baseline for complex questions, T5-Small for simple ones

**Medium Priority**
- Named-entity-aware sentence filtering to improve Who/Where accuracy
- Fine-tune T5 on HPC domain documents for better in-domain summarization

**Long-term**
- Replace word-overlap sentence selection with semantic similarity (e.g., cosine over sentence embeddings)
- Explore extractive + abstractive hybrid compression

---

## Citation

If you use this work, please cite:

```bibtex
@misc{rag-token-optimization-2026,
  title   = {Efficient Token Optimization in Retrieval-Augmented Generation Pipelines for HPC},
  author  = {Rasel & Samia},
  year    = {2026},
  school  = {Iowa State University},
  note    = {CS 6250 Course Project}
}
```
