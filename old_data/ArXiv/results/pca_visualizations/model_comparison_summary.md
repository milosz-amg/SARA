# ArXiv Embedding Models - Evaluation Results

**Total models evaluated:** 9
**Total papers:** 2397

## Ranking by Main Category Purity

| Rank | Model | Subcategory Purity | Main Category Purity | Silhouette Score | Clusters |
|------|-------|-------------------|---------------------|-----------------|----------|
| 1 | BAAI/bge-base-en-v1.5 | 19.02% | 70.55% | 0.3457 | 118 |
| 2 | BAAI/bge-large-en-v1.5 | 18.19% | 69.63% | 0.3468 | 118 |
| 3 | nomic-ai/nomic-embed-text-v1.5 | 17.52% | 69.50% | 0.3427 | 118 |
| 4 | allenai-specter | 18.52% | 68.88% | 0.3461 | 118 |
| 5 | Qwen/Qwen3-Embedding-0.6B | 17.31% | 68.09% | 0.3458 | 118 |
| 6 | sentence-transformers/all-mpnet-base-v2 | 18.48% | 68.09% | 0.3469 | 118 |
| 7 | Qwen/Qwen3-Embedding-4B | 17.48% | 67.75% | 0.3509 | 118 |
| 8 | Alibaba-NLP/gte-large-en-v1.5 | 17.52% | 67.63% | 0.3474 | 118 |
| 9 | intfloat/e5-mistral-7b-instruct | 14.94% | 65.08% | 0.3406 | 118 |

## Best Model

**BAAI/bge-base-en-v1.5**

- Main Category Purity: **70.55%**
- Subcategory Purity: 19.02%
- Silhouette Score: 0.3457
- Embedding Dimension: 768
- Optimal Clusters: 118

## Notes

- **Subcategory Purity**: Measured on 118 ArXiv subcategories (cs.AI, cs.LG, math.ST, etc.)
- **Main Category Purity**: Measured on ~10 main categories (cs, math, physics, etc.)
- **Silhouette Score**: Cluster quality metric (-1 to 1, higher is better)

Lower subcategory purity is expected due to the large number of fine-grained categories.