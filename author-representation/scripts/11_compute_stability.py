#!/usr/bin/env python3
"""
Compute Rolewski-style centroid stability for each WMI author.

Definition (Rolewski 2024, sec. 1.4.3):
    stab(k) = sim(c_S, c_all)
    c_S    = centroid of random subset S of size k
    c_all  = centroid of all author's papers

We compute stab(5) — mean cosine similarity between centroid of 5 random
papers and the full centroid, averaged over N=100 random draws.

Threshold s_min = 0.95 marks the boundary between stable and ambiguous
profiles (Rolewski's "akceptowalne odchylenie").
"""

import numpy as np
import pandas as pd
from pathlib import Path

# Paths
TITLES_CSV    = "data/titles_with_abstracts.csv"
SCIENTISTS_CSV = "data/scientists_with_identifiers.csv"
EMBEDDINGS_NPY = "data/paper_embeddings_cosent.npy"
OUTPUT_CSV    = "data/stability_scores.csv"

# Stability config (Rolewski-style)
SAMPLE_SIZE = 5      # k in stab(k)
N_REPEATS   = 100    # number of random draws to average over
SEED        = 42

# Load data
print("Loading data...")
titles_df     = pd.read_csv(TITLES_CSV)
scientists_df = pd.read_csv(SCIENTISTS_CSV)
scientists_df = scientists_df[scientists_df["orcid"].isin(titles_df["main_author_orcid"])]

embeddings = np.load(EMBEDDINGS_NPY)
paper_orcids = titles_df["main_author_orcid"].values

print(f"Authors: {len(scientists_df)}, Papers: {len(titles_df)}, Embeddings: {embeddings.shape}")

# Per-author stability
rng = np.random.default_rng(SEED)

records = []
for _, row in scientists_df.iterrows():
    orcid = row["orcid"]
    name  = row["full_name"]

    mask = paper_orcids == orcid
    author_embs = embeddings[mask]
    n_papers = len(author_embs)

    # Full centroid (normalized)
    c_all = author_embs.mean(axis=0)
    c_all = c_all / np.linalg.norm(c_all)

    # stab(5) — only meaningful if n >= 5
    if n_papers >= SAMPLE_SIZE:
        sims = []
        for _ in range(N_REPEATS):
            idx = rng.choice(n_papers, size=SAMPLE_SIZE, replace=False)
            c_s = author_embs[idx].mean(axis=0)
            c_s = c_s / np.linalg.norm(c_s)
            sims.append(float(c_s @ c_all))
        stab5_mean = np.mean(sims)
        stab5_std  = np.std(sims)
        stab5_min  = np.min(sims)
    else:
        stab5_mean = np.nan
        stab5_std  = np.nan
        stab5_min  = np.nan

    records.append({
        "orcid": orcid,
        "name": name,
        "n_papers": n_papers,
        "stab5_mean": stab5_mean,
        "stab5_std": stab5_std,
        "stab5_min": stab5_min,
    })

df = pd.DataFrame(records)
df = df.sort_values("n_papers", ascending=False).reset_index(drop=True)
df.to_csv(OUTPUT_CSV, index=False)

print(f"\nSaved: {OUTPUT_CSV}")
print(f"\nDistribution of stab5_mean (n>=5):")
valid = df.dropna(subset=["stab5_mean"])
print(f"  Authors evaluated: {len(valid)} (skipped {len(df) - len(valid)} with <{SAMPLE_SIZE} papers)")
print(f"  Mean:   {valid['stab5_mean'].mean():.4f}")
print(f"  Median: {valid['stab5_mean'].median():.4f}")
print(f"  Std:    {valid['stab5_mean'].std():.4f}")
print(f"\nThreshold breakdown (Rolewski thresholds):")
print(f"  stab5 >= 0.99 (high fidelity):        {(valid['stab5_mean'] >= 0.99).sum()} ({(valid['stab5_mean'] >= 0.99).mean()*100:.1f}%)")
print(f"  stab5 >= 0.95 (stable):               {(valid['stab5_mean'] >= 0.95).sum()} ({(valid['stab5_mean'] >= 0.95).mean()*100:.1f}%)")
print(f"  stab5 < 0.95  (ambiguous/multimodal): {(valid['stab5_mean'] < 0.95).sum()} ({(valid['stab5_mean'] < 0.95).mean()*100:.1f}%)")
