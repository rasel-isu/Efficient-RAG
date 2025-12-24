# RAG Experiments Comparative Analysis

*Generated: 2025-12-11 19:41:41*

---

## 📊 Executive Summary

**Best Overall Performance:** baseline_rag
- Exact Match: 58.28%
- F1 Score: 0.7040
- BERTScore F1: 0.9474

**Most Token Efficient:** used google/flan-t5-base for to retrieved docs chunks summary
- Avg Tokens/Question: 252.2
- Total Tokens: 231,525

---

## 1. Overall Performance Comparison

| Experiment                                                     |   Total Questions |   Exact Match (%) |   Partial Match (%) |   Mean F1 |   Median F1 |   Std F1 |
|:---------------------------------------------------------------|------------------:|------------------:|--------------------:|----------:|------------:|---------:|
| baseline_rag                                                   |               918 |           58.2789 |             71.8954 |    0.7040 |      1.0000 |   0.3994 |
| used google/flan-t5-small for to retrieved docs chunks summary |               918 |           55.3377 |             67.9739 |    0.6401 |      1.0000 |   0.4396 |
| used google/flan-t5-base for to retrieved docs chunks summary  |               918 |           53.7037 |             67.2113 |    0.6331 |      1.0000 |   0.4397 |
| used google/flan-t5-large for to retrieved docs chunks summary |               918 |           52.8322 |             64.8148 |    0.6140 |      1.0000 |   0.4491 |

### Key Insights:
- ✅ **baseline_rag** achieves the highest F1 score (0.7040)
- 📊 Performance spread: 0.6140 to 0.7040 (Δ 0.0900)

---

## 2. Advanced Metrics Comparison

| Experiment                                                     |   BERTScore Precision |   BERTScore Recall |   BERTScore F1 |   ROUGE-1 |   ROUGE-2 |   ROUGE-L |   Semantic Similarity |
|:---------------------------------------------------------------|----------------------:|-------------------:|---------------:|----------:|----------:|----------:|----------------------:|
| baseline_rag                                                   |                0.9465 |             0.9490 |         0.9474 |    0.7242 |    0.1657 |    0.7192 |                0.7959 |
| used google/flan-t5-small for to retrieved docs chunks summary |                0.9405 |             0.9381 |         0.9389 |    0.6530 |    0.1269 |    0.6509 |                0.7480 |
| used google/flan-t5-base for to retrieved docs chunks summary  |                0.9419 |             0.9383 |         0.9397 |    0.6487 |    0.1334 |    0.6461 |                0.7486 |
| used google/flan-t5-large for to retrieved docs chunks summary |                0.9393 |             0.9337 |         0.9360 |    0.6259 |    0.1166 |    0.6232 |                0.7120 |

### Key Insights:
- 🤖 **baseline_rag** has highest BERTScore F1 (0.9474), indicating strong semantic similarity
- 📝 **baseline_rag** achieves best ROUGE-L (0.7192)
- 📊 All experiments show strong BERTScore (>0.93), indicating high semantic quality

---

## 3. Token Efficiency Comparison

| Experiment                                                     |   Avg Prompt Tokens |   Avg Completion Tokens |   Avg Total Tokens |   Total Tokens Used |   Median Prompt |   Median Completion |   Estimated Cost ($) |
|:---------------------------------------------------------------|--------------------:|------------------------:|-------------------:|--------------------:|----------------:|--------------------:|---------------------:|
| used google/flan-t5-base for to retrieved docs chunks summary  |              248.72 |                    3.48 |             252.21 |              231525 |          244.00 |                1.00 |                 0.12 |
| used google/flan-t5-small for to retrieved docs chunks summary |              258.24 |                    3.83 |             262.07 |              240580 |          252.00 |                1.00 |                 0.12 |
| used google/flan-t5-large for to retrieved docs chunks summary |              259.21 |                    3.86 |             263.06 |              241491 |          255.00 |                1.00 |                 0.12 |
| baseline_rag                                                   |              653.31 |                    4.89 |             658.19 |              604223 |          646.50 |                1.00 |                 0.30 |

### Key Insights:
- 💰 **used google/flan-t5-base for to retrieved docs chunks summary** is most efficient (231,525 tokens)
- 📉 Token savings: 372,698 tokens (61.7% reduction) compared to baseline
- 💵 Cost savings: $0.1158 vs $0.3021

---

## 4. Performance by Question Type

| Question Type   | baseline_rag   | t5-small      | t5-base       | t5-large      |
|:----------------|:---------------|:--------------|:--------------|:--------------|
| YES NO          | 89.8% / 0.899  | 85.2% / 0.853 | 83.6% / 0.836 | 85.7% / 0.857 |
| WHAT            | 28.1% / 0.489  | 26.7% / 0.409 | 29.0% / 0.434 | 24.0% / 0.391 |
| WHO             | 40.7% / 0.654  | 29.6% / 0.510 | 31.5% / 0.548 | 29.6% / 0.516 |
| WHEN            | 53.7% / 0.686  | 53.7% / 0.615 | 41.5% / 0.539 | 31.7% / 0.410 |
| WHERE           | 34.4% / 0.604  | 28.1% / 0.549 | 18.8% / 0.516 | 25.0% / 0.470 |
| HOW             | 22.2% / 0.446  | 25.4% / 0.386 | 17.5% / 0.354 | 15.9% / 0.303 |
| WHY             | 4.0% / 0.373   | 0.0% / 0.196  | 4.0% / 0.214  | 4.0% / 0.212  |
| WHICH           | 45.5% / 0.624  | 36.4% / 0.573 | 45.5% / 0.604 | 36.4% / 0.513 |
| OTHER           | 41.2% / 0.659  | 47.1% / 0.651 | 41.2% / 0.624 | 39.2% / 0.537 |

### Key Insights:
- ✅ **Consistently Strong**: YES_NO (86.1%)
- 🔴 **Consistently Weak**: WHAT (26.9%), WHERE (26.6%), HOW (20.2%), WHY (3.0%)
- 📊 Yes/No questions perform best across all experiments (>83%)
- ⚠️  Why questions remain challenging across all approaches (<5%)

---

## 5. Error Analysis

| Experiment                                                     |   Total Errors |   Error Rate (%) |   Correct Answers |   Accuracy (%) |
|:---------------------------------------------------------------|---------------:|-----------------:|------------------:|---------------:|
| baseline_rag                                                   |            383 |            41.72 |               535 |          58.28 |
| used google/flan-t5-small for to retrieved docs chunks summary |            410 |            44.66 |               508 |          55.34 |
| used google/flan-t5-base for to retrieved docs chunks summary  |            425 |            46.30 |               493 |          53.70 |
| used google/flan-t5-large for to retrieved docs chunks summary |            433 |            47.17 |               485 |          52.83 |

---

## 6. Recommendations

### Based on Performance:

1. **Use baseline_rag** for best overall accuracy
   - Achieves 58.3% exact match and 0.7040 F1 score

### Based on Efficiency:

2. **Consider used google/flan-t5-base for to retrieved docs chunks summary** for cost-sensitive applications
   - Uses only 252 tokens per question on average

### Based on Question Type Analysis:

3. **Improve Why/How questions** across all experiments:
   - Add chain-of-thought prompting
   - Increase completion token limits
   - Include reasoning examples in context

4. **Consider hybrid approach:**
   - Use baseline for Yes/No questions (90% accuracy)
   - Use summarization models for factual questions
   - Balance performance vs. cost based on use case

## 7. Conclusion

The comparative analysis of 4 RAG experiments reveals several key findings:

1. **baseline_rag** achieves the best overall performance with 0.7040 F1 score
2. **used google/flan-t5-base for to retrieved docs chunks summary** offers the best cost-efficiency with 231,525 total tokens
3. All experiments show strong performance on Yes/No questions (>83% accuracy)
4. Why questions remain consistently challenging across all approaches (<5% accuracy)
5. BERTScore metrics (>0.93) indicate high semantic quality across all experiments

### Trade-offs:
- **Baseline RAG** provides the best accuracy but uses 604,223 tokens
- **Summarization approaches** reduce token usage by 60%+ but sacrifice 2-5% accuracy
- The choice between approaches should be guided by specific use case requirements (accuracy vs. cost)

### Next Steps:
1. Investigate hybrid approaches combining strengths of each method
2. Focus improvement efforts on Why/How question types
3. Consider fine-tuning for domain-specific performance
4. Implement context-aware routing (use different strategies for different question types)