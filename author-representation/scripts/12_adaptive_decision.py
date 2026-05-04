#!/usr/bin/env python3
"""
Compute per-author adaptive policy: SINGLE / MULTI / LOW_CONF / AMBIGUOUS.

Combines two diagnostic signals:
  - stability  s5 = mean cosine similarity stab(5) over 100 random subsets
                   of size 5 (Rolewski 2024, sec. 1.4.3)
  - silhouette best silhouette score across k=2..5 K-Means clusterings
                   (our multi-cluster approach)

Decision tree:
  n < 5                                              -> LOW_CONF
  n >= 5 AND silhouette_max >= 0.15                  -> MULTI       (k centroids)
  n >= 5 AND silhouette_max <  0.15 AND s5 >= P25    -> SINGLE      (1 centroid)
  n >= 5 AND silhouette_max <  0.15 AND s5 <  P25    -> AMBIGUOUS   (1 centroid + flag)

Rationale:
  - silhouette_max is the primary signal: high score => meaningful sub-clusters
  - stability is the confidence signal: when no clusters but profile unstable,
    we mark the author as ambiguous instead of declaring them "single-topic"
  - P25 (per-corpus 25th percentile of stab5_mean) replaces Rolewski's fixed
    0.95 threshold, which is too lax for our fine-tuned BGE embeddings
"""

import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# Paths
TITLES_CSV     = "data/titles_with_abstracts.csv"
SCIENTISTS_CSV = "data/scientists_with_identifiers.csv"
EMBEDDINGS_NPY = "data/paper_embeddings_cosent.npy"
STABILITY_CSV  = "data/stability_scores.csv"
OUTPUT_CSV     = "data/policy.csv"

# Thresholds
MIN_PAPERS = 5
MAX_K      = 5
SIL_THRESHOLD = 0.15

# Load
print("Loading data...")
titles_df     = pd.read_csv(TITLES_CSV)
scientists_df = pd.read_csv(SCIENTISTS_CSV)
scientists_df = scientists_df[scientists_df["orcid"].isin(titles_df["main_author_orcid"])]
embeddings = np.load(EMBEDDINGS_NPY)
paper_orcids = titles_df["main_author_orcid"].values
stability_df = pd.read_csv(STABILITY_CSV).set_index("orcid")

# Compute corpus-relative threshold for stability
valid_stab = stability_df["stab5_mean"].dropna()
P25 = float(np.percentile(valid_stab, 25))
print(f"Stability P25 (corpus-relative threshold): {P25:.4f}")
print(f"Silhouette threshold (multi-cluster): {SIL_THRESHOLD}")
print(f"Min papers for clustering: {MIN_PAPERS}\n")


def best_silhouette(embs):
    """Return (best_k, best_silhouette) for K-Means with k=2..MAX_K."""
    n = len(embs)
    max_k = min(MAX_K, n - 1)
    if max_k < 2:
        return 1, -1.0

    best_k, best_sil = 1, -1.0
    for k in range(2, max_k + 1):
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = km.fit_predict(embs)
        if len(set(labels)) < 2:
            continue
        sil = silhouette_score(embs, labels)
        if sil > best_sil:
            best_sil = sil
            best_k = k
    return best_k, best_sil


# Decide per author
records = []
for _, row in scientists_df.iterrows():
    orcid = row["orcid"]
    name  = row["full_name"]

    mask = paper_orcids == orcid
    author_embs = embeddings[mask]
    n_papers = len(author_embs)

    # Stability lookup
    if orcid in stability_df.index:
        s5_mean = stability_df.loc[orcid, "stab5_mean"]
        s5_min  = stability_df.loc[orcid, "stab5_min"]
    else:
        s5_mean = np.nan
        s5_min  = np.nan

    # Silhouette
    if n_papers >= MIN_PAPERS:
        best_k, sil_max = best_silhouette(author_embs)
    else:
        best_k, sil_max = 1, np.nan

    # Decision
    if n_papers < MIN_PAPERS:
        decision = "LOW_CONF"
        n_points = 1
    elif sil_max >= SIL_THRESHOLD:
        decision = "MULTI"
        n_points = best_k
    elif s5_mean >= P25:
        decision = "SINGLE"
        n_points = 1
    else:
        decision = "AMBIGUOUS"
        n_points = 1

    records.append({
        "orcid": orcid,
        "name": name,
        "n_papers": n_papers,
        "stab5_mean": s5_mean,
        "stab5_min": s5_min,
        "silhouette_max": sil_max,
        "best_k": best_k,
        "decision": decision,
        "n_points": n_points,
    })

policy = pd.DataFrame(records)
policy.to_csv(OUTPUT_CSV, index=False)

print(f"Saved: {OUTPUT_CSV}\n")

# Distribution
total = len(policy)
print("Policy distribution:")
for decision in ["SINGLE", "MULTI", "LOW_CONF", "AMBIGUOUS"]:
    sub = policy[policy["decision"] == decision]
    n = len(sub)
    pct = n / total * 100
    pts = sub["n_points"].sum()
    print(f"  {decision:10s}: {n:4d} authors ({pct:5.1f}%)  -> {pts} points")

print(f"\nTotal authors: {total}")
print(f"Total points (adaptive): {policy['n_points'].sum()}")
print(f"Compare:")
print(f"  baseline   = {total} (1 point per author)")
print(f"  aggressive = {sum(max(2, r['best_k']) if r['n_papers'] >= MIN_PAPERS and r['best_k'] > 1 else 1 for _, r in policy.iterrows())} (k>1 for all eligible)")
