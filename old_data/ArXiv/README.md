# ArXiv Paper Analysis System

A standalone system for evaluating multiple embedding models using ArXiv scientific papers, PCA clustering, and validation metrics.

## Overview

This project tests 8+ embedding models from the MTEB benchmark on 50,000 ArXiv papers across 5 major categories to determine which model best captures semantic relationships in scientific literature.

### Key Features

- **Multi-Model Comparison**: Test multiple embedding models (BGE, E5, Specter, etc.)
- **Large-Scale Analysis**: Process 50k papers across cs, physics, math, biology, and statistics
- **PCA Visualization**: Interactive 2D and 3D cluster visualizations
- **Validation Metrics**: Category purity and silhouette scores to rank model performance
- **Google Colab Support**: Run GPU-intensive embedding generation in Colab

## Directory Structure

```
ArXiv/
├── README.md                          # This file
├── requirements.txt                   # Python dependencies
├── config.py                          # Configuration constants
│
├── data/                              # Data storage
│   ├── raw/                           # Raw ArXiv API responses (JSON)
│   ├── arxiv_papers.db                # SQLite database
│   └── embeddings/                    # Per-model embedding storage
│
├── scripts/                           # Core scripts
│   ├── 01_fetch_arxiv_papers.py       # Fetch from ArXiv API
│   ├── 02_create_database.py          # Create SQLite DB
│   ├── 03_generate_embeddings.py      # Generate embeddings (TODO)
│   ├── 04_pca_clustering.py           # PCA + clustering (TODO)
│   ├── 05_evaluate_models.py          # Calculate metrics (TODO)
│   └── utils/
│       ├── arxiv_client.py            # ArXiv API wrapper
│       ├── embedding_utils.py         # Embedding helpers (TODO)
│       └── visualization.py           # Visualization helpers (TODO)
│
├── notebooks/                         # Jupyter/Colab notebooks
│   └── (TODO)
│
└── results/                           # Generated outputs
    ├── pca_visualizations/            # PCA plots
    ├── cluster_reports/               # Statistics
    └── model_comparison_report.json   # Final rankings
```

## Quick Start

### 1. Installation

```bash
cd ArXiv
pip install -r requirements.txt
```

### 2. Fetch ArXiv Papers

Fetch papers from ArXiv API (note: this takes ~40 hours due to 3-second rate limits):

```bash
# Fetch all configured categories (50k papers total)
python scripts/01_fetch_arxiv_papers.py --all-categories

# Or fetch a single category
python scripts/01_fetch_arxiv_papers.py --category cs --limit 10000
```

Papers are saved to `data/raw/papers_{category}.json`

### 3. Create Database

Create SQLite database from raw JSON files:

```bash
python scripts/02_create_database.py
```

This creates `data/arxiv_papers.db` and prints a validation report.

### 4. Generate Embeddings

(TODO: Implementation in progress)

```bash
# Generate embeddings for a specific model
python scripts/03_generate_embeddings.py --model BAAI/bge-large-en-v1.5

# Generate embeddings for all models
python scripts/03_generate_embeddings.py --all-models
```

### 5. Run PCA Clustering

(TODO: Implementation in progress)

```bash
# Run PCA and clustering for a specific model
python scripts/04_pca_clustering.py --model BAAI/bge-large-en-v1.5

# Run for all models
python scripts/04_pca_clustering.py --all-models
```

### 6. Evaluate Models

(TODO: Implementation in progress)

```bash
# Generate comparison report
python scripts/05_evaluate_models.py
```

## Configuration

Edit `config.py` to customize:

- **Target categories and counts**: `ARXIV_CATEGORIES`
- **Embedding models to test**: `EMBEDDING_MODELS`
- **Batch sizes**: `BATCH_SIZE_BY_DIM`
- **Clustering parameters**: `MIN_CLUSTERS`, `MAX_CLUSTERS`
- **Visualization settings**: `VIZ_WIDTH`, `VIZ_HEIGHT`, etc.

## Embedding Models

The system tests the following models from the MTEB benchmark:

| Model | Dimension | Type |
|-------|-----------|------|
| BAAI/bge-large-en-v1.5 | 1024 | SOTA general purpose |
| Alibaba-NLP/gte-large-en-v1.5 | 1024 | Strong retrieval |
| Qwen/Qwen3-Embedding-0.6B | 1024 | Your baseline |
| intfloat/e5-mistral-7b-instruct | 4096 | Large instruction-tuned |
| sentence-transformers/all-mpnet-base-v2 | 768 | Balanced baseline |
| BAAI/bge-base-en-v1.5 | 768 | Smaller BGE variant |
| nomic-ai/nomic-embed-text-v1.5 | 768 | Fast inference |
| allenai-specter | 768 | Scientific paper specialist |

## Validation Metrics

### Category Purity

Measures what percentage of papers in each cluster belong to the dominant ArXiv category.

```
Purity = (# papers in dominant category) / (total papers in cluster)
Overall Purity = average across all clusters
```

Higher is better. Range: 0-100%

### Silhouette Score

Measures how well-separated the clusters are.

```
Score = (distance to nearest other cluster - distance within cluster) / max(both)
```

Higher is better. Range: -1 to 1

### Davies-Bouldin Index

Measures cluster separation (lower is better).

### PCA Variance Explained

Percentage of information retained after dimensionality reduction.

## Database Schema

### papers table

```sql
CREATE TABLE papers (
    id TEXT PRIMARY KEY,              -- ArXiv ID
    title TEXT NOT NULL,
    abstract TEXT,
    authors TEXT,                     -- JSON array
    categories TEXT NOT NULL,         -- JSON array
    primary_category TEXT NOT NULL,   -- Main category
    published_date TEXT,
    updated_date TEXT,
    doi TEXT,
    arxiv_url TEXT,
    pdf_url TEXT,
    comment TEXT,
    journal_ref TEXT,
    created_at TIMESTAMP
);
```

### embedding_models table

```sql
CREATE TABLE embedding_models (
    model_name TEXT PRIMARY KEY,
    dimension INTEGER NOT NULL,
    generated_at TIMESTAMP,
    num_papers_embedded INTEGER,
    avg_time_per_paper REAL,
    notes TEXT
);
```

### evaluation_results table

```sql
CREATE TABLE evaluation_results (
    id INTEGER PRIMARY KEY,
    model_name TEXT NOT NULL,
    category_purity_overall REAL,
    silhouette_score REAL,
    davies_bouldin_score REAL,
    num_clusters INTEGER,
    pca_variance_explained REAL,
    evaluation_date TIMESTAMP,
    detailed_results TEXT            -- JSON
);
```

## Google Colab Usage

For GPU-accelerated embedding generation:

1. Upload database to Google Drive
2. Open Colab notebook (TODO: create notebook)
3. Mount Drive and install requirements
4. Run embedding generation scripts
5. Download embeddings back to local

## API Rate Limits

ArXiv API requires **3 seconds** between requests. Fetching 50k papers takes approximately:

```
50,000 papers / 1,000 per request = 50 requests
50 requests × 3 seconds = 150 seconds per category
5 categories × 150 seconds = 750 seconds = ~12.5 minutes minimum
```

However, with pagination and processing time, expect **several hours** for the full dataset.

## Expected Results

Based on MTEB benchmarks, expected model ranking by category purity:

1. **BAAI/bge-large-en-v1.5** - SOTA on MTEB (~80-85% purity)
2. **allenai-specter** - Scientific paper specialist (~78-83%)
3. **Alibaba-NLP/gte-large-en-v1.5** - Strong retrieval (~76-82%)
4. **intfloat/e5-mistral-7b** - High capacity (~75-80%)
5. **sentence-transformers/all-mpnet-base-v2** - Solid baseline (~70-75%)
6. **nomic-ai/nomic-embed-text-v1.5** - Fast but less accurate (~68-73%)
7. **BAAI/bge-base-en-v1.5** - Smaller variant (~67-72%)
8. **Qwen/Qwen3-Embedding-0.6B** - Your baseline (~65-70%)

## Troubleshooting

### "No papers fetched"

- Check internet connection
- Verify ArXiv API is accessible: `curl http://export.arxiv.org/api/query?search_query=cat:cs&max_results=1`
- Check rate limit delays in `config.py`

### "Database locked"

- Close any other connections to the database
- Ensure only one script accesses the database at a time

### GPU out of memory

- Reduce batch size in `config.py`: `BATCH_SIZE_BY_DIM`
- Use Google Colab with larger GPU (T4/A100)
- Process models sequentially rather than in parallel

### Slow fetching

- ArXiv requires 3-second delays (API requirement)
- Run fetching in background or overnight
- Consider using pre-downloaded datasets (Kaggle)

## Development Status

- [x] Project structure
- [x] Configuration system
- [x] ArXiv API client
- [x] Data fetching script
- [x] Database creation script
- [ ] Embedding generation script
- [ ] Embedding utilities
- [ ] PCA clustering script
- [ ] Visualization utilities
- [ ] Evaluation script
- [ ] Google Colab notebook
- [ ] Comparison report generation

## Contributing

This is part of the SARA project. For questions or contributions, please refer to the main SARA repository.

## License

Same as SARA project.

## References

- ArXiv API Documentation: https://arxiv.org/help/api/index
- MTEB Benchmark: https://huggingface.co/spaces/mteb/leaderboard
- Sentence Transformers: https://www.sbert.net/
- BGE Embeddings: https://github.com/FlagOpen/FlagEmbedding
