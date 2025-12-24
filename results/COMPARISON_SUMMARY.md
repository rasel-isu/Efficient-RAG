# RAG Experiments Comparison - Quick Summary

## 🏆 Winner by Category

| Category | Winner | Score |
|----------|--------|-------|
| **Best Accuracy** | Baseline RAG | 58.28% EM / 0.704 F1 |
| **Most Efficient** | T5-Base Summary | 231,525 tokens (62% savings) |
| **Best BERTScore** | Baseline RAG | 0.9474 F1 |
| **Best Balance** | T5-Small Summary | 55.34% EM / 0.640 F1 @ 240K tokens |

---

## 📊 Overall Performance Comparison

| Experiment | Exact Match | F1 Score | Partial Match | BERTScore | Tokens Used |
|------------|-------------|----------|---------------|-----------|-------------|
| **Baseline RAG** | **58.28%** | **0.7040** | **71.90%** | **0.9474** | 604,223 |
| T5-Small Summary | 55.34% | 0.6401 | 67.97% | 0.9389 | **240,580** ⭐ |
| T5-Base Summary | 53.70% | 0.6331 | 67.21% | 0.9397 | **231,525** ⭐⭐ |
| T5-Large Summary | 52.83% | 0.6140 | 64.81% | 0.9360 | 241,491 |

**Key Insight:** Baseline is most accurate but uses 2.6x more tokens. T5-Base offers best cost-efficiency.

---

## 💰 Cost-Benefit Analysis

| Experiment | Accuracy | Cost/918 Questions | Efficiency Score* |
|------------|----------|-------------------|-------------------|
| Baseline RAG | 58.28% | $0.30 | **1.94** |
| T5-Small | 55.34% | $0.12 | **4.61** ⭐⭐ |
| T5-Base | 53.70% | $0.12 | **4.48** ⭐ |
| T5-Large | 52.83% | $0.12 | 4.39 |

*Efficiency Score = Accuracy / Cost (higher is better)

**Key Insight:** T5-Small provides best cost-performance ratio - 95% of baseline accuracy at 40% of the cost.

---

## 📈 Performance by Question Type

| Question Type | Baseline | T5-Small | T5-Base | T5-Large | Average |
|---------------|----------|----------|---------|----------|---------|
| **Yes/No** (420) | 89.8% ✅ | 85.2% | 83.6% | 85.7% | 86.1% |
| **When** (41) | 53.7% | 53.7% | 41.5% | 31.7% | 45.2% |
| **Which** (11) | 45.5% | 36.4% | 45.5% | 36.4% | 40.9% |
| **Other** (51) | 41.2% | 47.1% | 41.2% | 39.2% | 42.2% |
| **Who** (54) | 40.7% | 29.6% | 31.5% | 29.6% | 32.9% |
| **Where** (32) | 34.4% | 28.1% | 18.8% | 25.0% | 26.6% |
| **What** (221) | 28.1% 🔴 | 26.7% | 29.0% | 24.0% | 26.9% |
| **How** (63) | 22.2% 🔴 | 25.4% | 17.5% | 15.9% | 20.2% |
| **Why** (25) | 4.0% 🔴 | 0.0% | 4.0% | 4.0% | 3.0% |

**Key Insights:**
- ✅ All models excel at Yes/No questions (>83%)
- 🔴 All models struggle with Why questions (<5%)
- 📊 What questions (24% of dataset) show consistently low performance
- 🎯 Baseline outperforms on most question types

---

## 🔍 Advanced Metrics Comparison

### BERTScore (Semantic Similarity)

| Experiment | Precision | Recall | F1 | Interpretation |
|------------|-----------|--------|-----|----------------|
| Baseline | 0.9465 | 0.9490 | **0.9474** ⭐ | Excellent semantic match |
| T5-Small | 0.9405 | 0.9381 | 0.9389 | Excellent semantic match |
| T5-Base | 0.9419 | 0.9383 | 0.9397 | Excellent semantic match |
| T5-Large | 0.9393 | 0.9337 | 0.9360 | Excellent semantic match |

**Key Insight:** All models achieve >93% BERTScore, indicating high semantic quality despite exact match differences.

### ROUGE Scores (Text Overlap)

| Experiment | ROUGE-1 | ROUGE-2 | ROUGE-L |
|------------|---------|---------|---------|
| Baseline | **0.724** | **0.166** | **0.719** |
| T5-Small | 0.653 | 0.127 | 0.651 |
| T5-Base | 0.649 | 0.133 | 0.646 |
| T5-Large | 0.626 | 0.117 | 0.623 |

**Key Insight:** Baseline shows better lexical overlap with ground truth answers.

### Semantic Similarity (Cosine)

| Experiment | Mean Similarity | Std Dev | Min | Max |
|------------|----------------|---------|-----|-----|
| Baseline | **0.796** | 0.288 | -0.046 | 1.000 |
| T5-Small | 0.748 | 0.329 | -0.062 | 1.000 |
| T5-Base | 0.749 | 0.328 | -0.062 | 1.000 |
| T5-Large | 0.712 | 0.356 | -0.062 | 1.000 |

---

## 📉 Accuracy vs Cost Trade-off

```
Accuracy (%)
    60 |  ● Baseline (58.28%)
       |
    55 |     ● T5-Small (55.34%)
       |        ● T5-Base (53.70%)
    50 |           ● T5-Large (52.83%)
       |
       +--------------------------------
         $0.12   $0.20   $0.28   Cost

Best Balance: T5-Small
Best Accuracy: Baseline
Best Efficiency: T5-Base
```

---

## 🎯 Recommendations

### Choose Baseline RAG When:
- ✅ Accuracy is critical (58.28% vs 53-55% for T5)
- ✅ Cost is not a primary concern
- ✅ Best semantic quality needed (0.947 BERTScore)
- ✅ Need highest performance on all question types

### Choose T5-Small Summary When:
- ✅ Need balance of performance and cost
- ✅ 55% accuracy is acceptable (5% accuracy drop)
- ✅ Want 60% token reduction
- ✅ Best cost-performance ratio (4.61 efficiency score)

### Choose T5-Base Summary When:
- ✅ Minimizing cost is priority (62% token reduction)
- ✅ 54% accuracy is acceptable
- ✅ Need lowest token consumption
- ✅ Volume/scale is high

### Avoid T5-Large Summary:
- ❌ Lowest accuracy (52.83%)
- ❌ No cost benefit over T5-Small/Base
- ❌ No performance benefit over smaller models

---

## 🔧 Improvement Opportunities

### High Priority (All Models):
1. **Why Questions** (3% avg accuracy)
   - Add chain-of-thought prompting
   - Provide reasoning examples
   - Increase completion token limits

2. **What Questions** (27% avg accuracy, 24% of dataset)
   - Most common question type
   - Low performance across all models
   - High impact improvement opportunity

3. **How Questions** (20% avg accuracy)
   - Similar challenges to Why questions
   - Need more detailed explanations

### Medium Priority:
4. **Who/Where Questions** (30-33% avg)
   - Named entity recognition improvements
   - Better context retrieval

### Low Priority:
5. **Yes/No Questions** (86% avg)
   - Already performing well
   - Maintain current approach

---

## 💡 Key Takeaways

1. **Performance-Cost Trade-off Exists:**
   - Baseline: Best accuracy but 2.6x more tokens
   - T5 models: 60% cheaper but 3-5% less accurate

2. **Semantic Quality Remains High:**
   - All models >0.93 BERTScore
   - Answers are semantically correct even when not exact matches

3. **Question Type Variability:**
   - Yes/No: 86% average (strong across all models)
   - Why: 3% average (critical weakness)
   - 9x performance difference between best and worst types

4. **T5 Model Size Paradox:**
   - Larger T5 models don't perform better
   - T5-Small outperforms T5-Large
   - Suggests summarization may lose important details

5. **Practical Recommendation:**
   - **Use T5-Small** for most applications (best balance)
   - **Use Baseline** only when accuracy is critical
   - **Avoid T5-Large** (no benefits)
   - Consider **hybrid approach**: Baseline for complex questions, T5-Small for simple ones

---

## 📊 Statistical Summary

| Metric | Baseline | T5-Small | T5-Base | T5-Large |
|--------|----------|----------|---------|----------|
| **Correct Answers** | 535 / 918 | 508 / 918 | 493 / 918 | 485 / 918 |
| **Error Rate** | 41.7% | 44.7% | 46.3% | 47.2% |
| **F1 Std Dev** | 0.399 | 0.440 | 0.440 | 0.449 |
| **Token Efficiency** | 1.00x | **2.51x** ⭐ | **2.61x** ⭐⭐ | 2.50x |

---

## 🎓 Conclusion

The comparative analysis reveals clear trade-offs:

- **Baseline RAG** remains the accuracy leader (58.28%) but at significant cost (604K tokens)
- **T5-Base** offers the best token efficiency (62% reduction) with modest accuracy impact (5% drop)
- **T5-Small** provides the optimal balance with 4.61 efficiency score
- **T5-Large** underperforms smaller models, suggesting over-compression

**Winner:** Depends on your priorities
- **Accuracy-critical:** Baseline RAG
- **Cost-sensitive:** T5-Base Summary  
- **Balanced:** T5-Small Summary ⭐ **Recommended**

All models struggle with Why/What/How questions, presenting a clear improvement opportunity regardless of which approach is chosen.
