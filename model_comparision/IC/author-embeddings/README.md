# Author Embeddings for Scientist Similarity

Part of the **SARA (Seek & Research)** project - computing semantic similarity between scientists based on their publication texts.

## Problem

Given a set of researchers and their publications, can we:
1. Create meaningful vector representations (embeddings) of each author?
2. Use these embeddings to find similar researchers?
3. Validate that the embeddings capture real research similarity?

## Approach

We aggregate each author's publication texts (titles + abstracts) and encode them using pre-trained sentence transformers:
- **SPECTER** (`allenai-specter`) - trained on scientific papers
- **MiniLM** (`all-MiniLM-L6-v2`) - general-purpose, smaller and faster

## Setup

```bash
# Create virtual environment (optional)
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or: venv\Scripts\activate  # Windows

# Install dependencies
pip install sentence-transformers scikit-learn pandas numpy matplotlib seaborn scipy tqdm plotly
```

## Data

| Dataset | Records | Source |
|---------|---------|--------|
| Scientists | 115 | UAM Research Portal |
| Publications | 3,440 | OpenAlex API |
| Co-author pairs | 57 | Extracted from publications |

Data files location:
- `data/titles_with_abstracts.csv` - publications with abstracts
- `data/scientists_with_identifiers.csv` - scientist profiles

## Notebooks

1. **`evaluation_and_comparison.ipynb`** - Main evaluation notebook
   - Co-author proximity test (do collaborators cluster together?)
   - SPECTER vs MiniLM comparison
   - Statistical significance tests

2. **`abstracts_vs_noabstracts.ipynb`** - Ablation study
   - Impact of including abstracts vs titles only
   - Semantic displacement visualization

## Quick Start

```bash
# Run the main evaluation
jupyter notebook evaluation_and_comparison.ipynb

# Or use the inference script directly
python inference.py
```

## Results

### Co-author Proximity Test

Both models show that co-authors are significantly closer in embedding space than random pairs:

| Model | Co-author Similarity | Random Similarity | Difference | Effect Size (d) | p-value |
|-------|---------------------|-------------------|------------|-----------------|---------|
| SPECTER | 0.787 | 0.626 | +0.161 | 1.53 | 4.9e-17 |
| MiniLM | 0.419 | 0.109 | +0.310 | 2.20 | 1.2e-19 |

**Key finding**: MiniLM achieves better separation despite being a general-purpose model.

### Model Agreement

Pearson correlation between model similarities: **r = 0.73**

## Files

```
author-embeddings/
├── README.md                        # This file
├── inference.py                     # Standalone inference script
├── evaluation_and_comparison.ipynb  # Main evaluation notebook
├── abstracts_vs_noabstracts.ipynb   # Titles vs abstracts comparison
├── report_outline.md                # Report structure (Polish)
├── evaluation_results.csv           # Saved results
├── data/
│   ├── titles_with_abstracts.csv    # Publications data
│   └── scientists_with_identifiers.csv  # Scientists data
├── *.png                            # Result visualizations
└── *.html                           # Interactive PCA plots
```

## References

- Reimers, N., & Gurevych, I. (2019). Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks. EMNLP.
- Cohan, A., et al. (2020). SPECTER: Document-level Representation Learning using Citation-informed Transformers. ACL.
- OpenAlex: https://openalex.org/

## Authors

SARA Project Team Members:
- Jakub Paszke
- Miłosz Rolewski


