# Fine-tuned BGE-base Evaluation

**Original model:** BAAI/bge-base-en-v1.5
**Fine-tuned model:** /home/jakub/Projekty/SARA/ArXiv/models/bge-base-cosent-finetuned/final
**Total papers:** 18880
**Evaluated at:** 2026-02-16 17:51

## Clustering Metrics

| k | Metric | Original | Fine-tuned | Diff |
|---|--------|----------|------------|------|
| 8 | main_category_purity | 65.76% | 70.49% | +4.74% |
| 8 | subcategory_purity | 9.52% | 10.09% | +0.57% |
| 8 | fuzzy_purity | 74.18% | 80.86% | +6.68% |
| 8 | silhouette_score | 0.3813 | 0.4378 | +0.0564 |
| 20 | main_category_purity | 67.13% | 72.60% | +5.47% |
| 20 | subcategory_purity | 11.43% | 12.93% | +1.50% |
| 20 | fuzzy_purity | 75.13% | 79.99% | +4.87% |
| 20 | silhouette_score | 0.3291 | 0.3494 | +0.0203 |
| 118 | main_category_purity | 68.07% | 73.76% | +5.69% |
| 118 | subcategory_purity | 13.38% | 16.17% | +2.79% |
| 118 | fuzzy_purity | 74.14% | 80.59% | +6.46% |
| 118 | silhouette_score | 0.3237 | 0.3272 | +0.0035 |

## Retrieval Metrics (Multi-Label)

| Metric | Original | Fine-tuned | Diff |
|--------|----------|------------|------|
| P@1 | 0.7400 | 0.7580 | +0.0180 |
| R@1 | 0.0015 | 0.0015 | +0.0000 |
| P@3 | 0.7065 | 0.7297 | +0.0232 |
| R@3 | 0.0043 | 0.0044 | +0.0001 |
| P@5 | 0.6840 | 0.7127 | +0.0287 |
| R@5 | 0.0069 | 0.0071 | +0.0002 |
| P@10 | 0.6537 | 0.6898 | +0.0361 |
| R@10 | 0.0129 | 0.0136 | +0.0006 |
| P@20 | 0.6245 | 0.6674 | +0.0429 |
| R@20 | 0.0244 | 0.0260 | +0.0016 |

## Notes

- **Fuzzy purity**: Paper is correct if it shares ANY category with cluster's dominant group
- **P@k/R@k**: A neighbor is relevant if it shares ANY category with the query (multi-label)
- P/R@k computed on 2000 sampled queries