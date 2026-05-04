# Publications Visualisation — Script Reference

Quick reference for running each script individually.
For the full automated pipeline, use `./run_visualisation.sh`.

---

## Directory layout

```
SARA/
├── wmii-data-collection/data/
│   ├── embeddings.npy
│   ├── embeddings_metadata.csv
│   └── scientists_with_identifiers.csv
└── publications-visualisation/
    ├── src/
    │   ├── common.py                  ← shared helpers, not run directly
    │   ├── find_optimal_k.py
    │   ├── analyse.py
    │   ├── choquet_composite.py
    │   ├── sugeno_composite.py
    │   ├── trimmed_composite.py
    │   ├── metric_correlations.py
    │   └── robustness_experiment.py
    └── output/                        ← all outputs land here
```

All commands below assume you're in `publications-visualisation/src/` with your venv active.

---

## Step 1 — `find_optimal_k.py`

Finds the optimal number of K-Means clusters using 6 methods (Elbow, Silhouette, Gap, Calinski-Harabasz, Davies-Bouldin, Stability) combined into a weighted composite score.

**Output:** `output/k_metrics.json`

```bash
# Default run (k = 4..12, reads embeddings from default path)
python find_optimal_k.py

# Custom k range
python find_optimal_k.py --k-min 5 --k-max 20

# Custom paths
python find_optimal_k.py \
    --embeddings ../../wmii-data-collection/data/embeddings.npy \
    --output     ../output/k_metrics.json \
    --k-min 4 \
    --k-max 16
```

| Argument | Default | Description |
|---|---|---|
| `--embeddings` | `../wmii-data-collection/data/embeddings.npy` | Path to embeddings `.npy` file |
| `--output` | `../output/k_metrics.json` | Where to write results |
| `--k-min` | `4` | Smallest k to test (≥ 2) |
| `--k-max` | `16` | Largest k to test |
| `--seed` | `42` | Random seed |

---

## Step 2 — `analyse.py`

Projects embeddings to 2D using 8 DR methods (UMAP, t-SNE, PaCMAP, Isomap, Spectral, PCA, PCA-8D, LDA) and evaluates each with 7 quality metrics. Produces the main JSON consumed by `vis_methods.html`.

**Output:** `output/vis_methods.json`

```bash
# Minimal — use k from step 1
python analyse.py --clusters 9

# With k-analysis JSON (enables k-analysis tab in the HTML viewer)
python analyse.py \
    --clusters   9 \
    --k-analysis ../output/k_metrics.json

# Full custom run
python analyse.py \
    --embeddings   ../../wmii-data-collection/data/embeddings.npy \
    --metadata     ../../wmii-data-collection/data/embeddings_metadata.csv \
    --scientists   ../../wmii-data-collection/data/scientists_with_identifiers.csv \
    --clusters     9 \
    --output       ../output/vis_methods.json \
    --k-analysis   ../output/k_metrics.json

# Skip t-SNE (much faster, useful for quick iteration)
python analyse.py --clusters 9 --no-tsne
```

| Argument | Default | Description |
|---|---|---|
| `--embeddings` | `../wmii-data-collection/data/embeddings.npy` | Path to embeddings |
| `--metadata` | `../wmii-data-collection/data/embeddings_metadata.csv` | Paper metadata CSV |
| `--scientists` | `../wmii-data-collection/data/scientists_with_identifiers.csv` | Optional — adds author names |
| `--clusters` | `9` | Number of K-Means clusters |
| `--output` | `../output/vis_methods.json` | Output JSON path |
| `--k-analysis` | `None` | Pass `k_metrics.json` to embed k-analysis in output |
| `--no-tsne` | off | Skip t-SNE projection |
| `--metric-sample` | `3440` | Points sampled for quality metric computation |

---

## Step 3 — `robustness_experiment.py`

Runs t-SNE, UMAP, and PaCMAP each with N independent random seeds and computes Choquet scores per run. Reports mean ± std, win counts, and Wilcoxon significance tests between method pairs.

**Requires:** `output/vis_methods.json` from step 2.
**Output:** `output/robustness_results.json`

```bash
# Default (10 seeds)
python robustness_experiment.py \
    --embeddings ../../wmii-data-collection/data/embeddings.npy \
    --input      ../output/vis_methods.json \
    --output     ../output/robustness_results.json

# Fewer seeds for a quick check
python robustness_experiment.py \
    --embeddings ../../wmii-data-collection/data/embeddings.npy \
    --input      ../output/vis_methods.json \
    --output     ../output/robustness_results.json \
    --seeds 5
```

| Argument | Default | Description |
|---|---|---|
| `--embeddings` | *(required)* | Path to embeddings `.npy` |
| `--input` | *(required)* | `vis_methods.json` from `analyse.py` (used for cluster labels) |
| `--output` | `robustness_results.json` | Output path |
| `--seeds` | `10` | Number of independent random seeds per method |
| `--sample` | `1500` | Points sampled for metric computation per run |

---

## Scoring scripts

The three scripts below all take the same `--input` argument pointing at `vis_methods.json`. They print results to stdout only — no output files are written.

### `choquet_composite.py`

Aggregates quality metrics using a validated 2-additive Choquet integral with pairwise synergy/redundancy interactions. The recommended composite scorer.

```bash
python choquet_composite.py --input ../output/vis_methods.json
```

Prints: capacity validation, singleton weights, pairwise interactions, per-method Choquet scores, ranking, and Shapley importance values.

---

### `sugeno_composite.py`

Aggregates using a Sugeno (bottleneck) integral with a λ-fuzzy measure. Tested across λ ∈ {0, −0.3, −0.6, −0.9, +0.3, +0.6} — useful for checking sensitivity to the redundancy/synergy assumption.

```bash
python sugeno_composite.py --input ../output/vis_methods.json

# Also print normalized metric values per method
python sugeno_composite.py --input ../output/vis_methods.json --show-metrics
```

| Argument | Default | Description |
|---|---|---|
| `--input` | *(required)* | `vis_methods.json` |
| `--show-metrics` | off | Print normalized metric table |

---

### `trimmed_composite.py`

Simple trimmed-mean baseline — drops the best and worst metric per method and averages the rest. Useful as a sanity check against the Choquet ranking.

```bash
python trimmed_composite.py --input ../output/vis_methods.json
```

---

### `metric_correlations.py`

Computes a Pearson correlation matrix between the 7 normalized quality metrics across all DR methods. Used to empirically validate the redundancy/synergy interactions assumed in the Choquet measure.

```bash
python metric_correlations.py --input ../output/vis_methods.json
```

Prints: normalized metric values per method, full correlation matrix (pairs with |r| > 0.7 marked), and a verdict on each assumed interaction.

---

## Typical full run (manual)

```bash
cd publications-visualisation/src

# 1. Find optimal k
python find_optimal_k.py --k-min 4 --k-max 16

# 2. Project & evaluate (use k from step 1, e.g. k=9)
python analyse.py --clusters 9 --k-analysis ../output/k_metrics.json

# 3. Robustness check
python robustness_experiment.py \
    --embeddings ../../wmii-data-collection/data/embeddings.npy \
    --input      ../output/vis_methods.json \
    --output     ../output/robustness_results.json

# 4. Composite scores
python choquet_composite.py  --input ../output/vis_methods.json
python sugeno_composite.py   --input ../output/vis_methods.json
python trimmed_composite.py  --input ../output/vis_methods.json

# 5. Validate metric interactions
python metric_correlations.py --input ../output/vis_methods.json
```

Or just run the full pipeline in one go:

```bash
cd publications-visualisation
./run_visualisation.sh --k-min 4 --k-max 16
```