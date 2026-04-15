"""
common.py
=========
Shared constants, data-loading helpers, and the STOPWORDS set used by
both find_optimal_k.py and analyse.py.

Import with:
    from common import (
        load_embeddings, load_metadata, name_clusters,
        DEFAULT_EMB, DEFAULT_META, DEFAULT_SCI, SEED, COLORS, STOPWORDS
    )
"""

import re
import warnings
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import normalize

warnings.filterwarnings("ignore")

# ── Defaults ──────────────────────────────────────────────────────────────────

DEFAULT_EMB  = "./data/embeddings.npy"
DEFAULT_META = "./data/embeddings_metadata.csv"
DEFAULT_SCI  = "./data/scientists_with_identifiers.csv"
DEFAULT_OUT  = "./wyniki/vis_methods.json"
DEFAULT_KA   = "./wyniki/k_metrics.json"

N_CLUSTERS   = 9
SEED         = 42
METRIC_N     = 3440

COLORS = [
    "#e6194b", "#3cb44b", "#4363d8", "#f58231",
    "#911eb4", "#42d4f4", "#f032e6", "#bfef45", "#FFFC3A",
]

STOPWORDS = {
    "of","the","and","in","for","a","on","to","with","an","by","from","is","are",
    "its","their","some","new","two","via","over","or","at","as","be","it","that",
    "this","which","not","we","using","based","under","between","about","into",
    "non","all","one","can","has","also","more","than","such","given","each",
    "show","results","paper","problem","method","class","set","case","three",
    "prove","study",
}

# ── Data helpers ──────────────────────────────────────────────────────────────

def load_embeddings(path: str) -> np.ndarray:
    """Load .npy embeddings and L2-normalise to unit sphere (cosine = dot product)."""
    arr = np.load(path).astype(np.float32)
    return normalize(arr)


def load_metadata(meta_path: str, sci_path: str) -> "pd.DataFrame | None":
    """
    Load paper metadata CSV and optionally join scientist names.
    Returns None if the metadata file is missing.
    """
    mp, sp = Path(meta_path), Path(sci_path)
    if not mp.exists():
        print(f"  [metadata] {meta_path} not found — continuing without metadata")
        return None
    meta = pd.read_csv(mp)
    if meta.columns[0] in ("", "Unnamed: 0"):
        meta = meta.rename(columns={meta.columns[0]: "_idx"})
    if sp.exists():
        sci = pd.read_csv(sp)
        meta = meta.merge(
            sci[["orcid", "full_name"]],
            left_on="main_author_orcid", right_on="orcid", how="left",
        )
    else:
        meta["full_name"] = None
    print(f"  [metadata] {len(meta)} rows loaded")
    return meta


def name_clusters(meta: "pd.DataFrame | None", labels: np.ndarray, n_clusters: int) -> dict:
    """
    Derive human-readable cluster names from the 3 most common non-stopword
    title words for each cluster.
    Falls back to "Cluster N" when metadata / titles are unavailable.
    """
    if meta is None or "title" not in meta.columns:
        return {i: f"Cluster {i}" for i in range(n_clusters)}
    names = {}
    for c in range(n_clusters):
        idxs   = np.where(labels == c)[0]
        titles = meta.iloc[idxs]["title"].dropna().str.lower()
        words  = [
            w
            for t in titles
            for w in re.findall(r"[a-z]+", t)
            if len(w) > 3 and w not in STOPWORDS
        ]
        top3 = [w for w, _ in Counter(words).most_common(3)]
        names[c] = ", ".join(top3) if top3 else f"Cluster {c}"
    return names