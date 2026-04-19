"""
Analyses the optimal number of K-Means clusters for a set of embeddings
using six complementary methods and writes the results to JSON.

Methods:
  1. Elbow / Inertia        - kink in SSE curve (2nd-derivative)
  2. Silhouette Score       - mean cohesion vs. separation  ↑ higher = better
  3. Gap Statistic          - vs. random uniform reference  ↑ higher = better
  4. Calinski-Harabasz (CH) - between/within variance ratio ↑ higher = better
  5. Davies-Bouldin (DB)    - average cluster similarity    ↓ lower  = better
  6. Stability              - bootstrap Jaccard consistency ↑ higher = better

WHY NOT SIMPLE MAJORITY VOTE?
  Gap, CH, and Stability are often monotonic - Gap always increases with k,
  CH always decreases, Stability always decreases.  A raw vote gives those
  methods permanent votes for k_max / k_min regardless of the data, drowning
  out the informative methods (Silhouette, Elbow, DB).

  Instead, all six scores are min-max normalised to [0,1] so each method
  contributes equally.  The composite is:
      0.30 * sil_norm
    + 0.25 * elbow_norm   (2nd-derivative spike)
    + 0.20 * db_norm      (lower DB = better, so inverted)
    + 0.15 * ch_norm
    + 0.10 * gap_norm
    + 0.00 * stability    (excluded - almost always monotone decreasing)

  The k with the highest composite wins.  Individual per-method picks are
  still reported for reference.

Usage
-----
  python find_optimal_k.py
  python find_optimal_k.py --k-min 5 --k-max 20
  python find_optimal_k.py --embeddings ../data/embeddings.npy --k-min 4 --k-max 16
"""

import argparse
import json
import warnings
from pathlib import Path

import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import (
    calinski_harabasz_score,
    davies_bouldin_score,
    silhouette_score,
)

from common import DEFAULT_EMB, DEFAULT_KA, SEED, load_embeddings

warnings.filterwarnings("ignore")


# Helpers

def _gap_statistic(
    pca50: np.ndarray,
    km: KMeans,
    k: int,
    n_refs: int = 8,
    seed: int = SEED,
) -> tuple:
    rng = np.random.default_rng(seed)
    mins, maxs = pca50.min(axis=0), pca50.max(axis=0)

    ref_log_inertias = []
    for _ in range(n_refs):
        ref = rng.uniform(mins, maxs, size=pca50.shape).astype(np.float32)
        ref_km = KMeans(n_clusters=k, random_state=int(rng.integers(1_000_000)), n_init=5)
        ref_km.fit(ref)
        ref_log_inertias.append(np.log(ref_km.inertia_ + 1e-12))

    log_ref = np.array(ref_log_inertias)
    gap     = float(log_ref.mean() - np.log(km.inertia_ + 1e-12))
    gap_std = float(log_ref.std() * np.sqrt(1 + 1 / n_refs))
    return round(gap, 4), round(gap_std, 4)


def _stability_analysis(
    emb: np.ndarray,
    k: int,
    n_runs: int = 6,
    subsample: float = 0.8,
    seed: int = SEED,
) -> float:
    rng   = np.random.default_rng(seed)
    n     = len(emb)
    sub_n = int(n * subsample)

    all_labels  = []
    all_indices = []

    for _ in range(n_runs):
        idx = rng.choice(n, sub_n, replace=False)
        km  = KMeans(n_clusters=k, random_state=int(rng.integers(1_000_000)), n_init=5)
        lab = km.fit_predict(emb[idx])
        all_labels.append(lab)
        all_indices.append(np.sort(idx))

    jaccard_scores = []
    for i in range(n_runs):
        for j in range(i + 1, n_runs):
            shared = np.intersect1d(all_indices[i], all_indices[j])
            if len(shared) < 10:
                continue
            li = all_labels[i][np.searchsorted(all_indices[i], shared)]
            lj = all_labels[j][np.searchsorted(all_indices[j], shared)]

            same_i = li[:, None] == li[None, :]
            same_j = lj[:, None] == lj[None, :]
            inter  = float((same_i & same_j).sum())
            union  = float((same_i | same_j).sum())
            jaccard_scores.append(inter / (union + 1e-12))

    return round(float(np.mean(jaccard_scores)) if jaccard_scores else 0.0, 4)


def _norm01(arr: np.ndarray) -> np.ndarray:
    """Min-max normalise to [0, 1]. Returns zeros if all values are equal."""
    mn, mx = arr.min(), arr.max()
    if mx - mn < 1e-12:
        return np.zeros_like(arr)
    return (arr - mn) / (mx - mn)


# Main analysis

def optimal_k_analysis(
    emb: np.ndarray,
    k_range: range = range(5, 21),
    seed: int = SEED,
) -> dict:
    rng = np.random.default_rng(seed)

    sidx    = rng.choice(len(emb), min(2000, len(emb)), replace=False)
    emb_sub = emb[sidx]

    print("  Computing PCA-50 for Gap reference space...")
    pca50 = PCA(n_components=min(50, emb.shape[1]), random_state=seed).fit_transform(emb)

    results = []

    header = (f"  {'k':>3}  {'Inertia':>10}  {'Silhouette':>10}  "
              f"{'Gap':>7}  {'Gap±':>6}  {'CH':>9}  {'DB':>7}  {'Stability':>9}")
    print(header)
    print("  " + "-" * (len(header) - 2))

    for k in k_range:
        km  = KMeans(n_clusters=k, random_state=seed, n_init=10)
        lab = km.fit_predict(emb)

        inertia   = float(km.inertia_)
        sil       = float(silhouette_score(emb_sub, lab[sidx], metric="cosine"))
        gap, gstd = _gap_statistic(pca50, km, k, n_refs=8, seed=seed)
        ch        = float(calinski_harabasz_score(emb, lab))
        db        = float(davies_bouldin_score(emb, lab))
        stability = _stability_analysis(emb, k, n_runs=6, subsample=0.8, seed=seed)

        row = {
            "k":                 k,
            "inertia":           round(inertia, 1),
            "silhouette":        round(sil, 4),
            "gap":               gap,
            "gap_std":           gstd,
            "calinski_harabasz": round(ch, 1),
            "davies_bouldin":    round(db, 4),
            "stability":         stability,
        }
        results.append(row)
        print(f"  {k:>3}  {inertia:>10.1f}  {sil:>10.4f}  {gap:>7.4f}  "
              f"{gstd:>6.4f}  {ch:>9.1f}  {db:>7.4f}  {stability:>9.4f}")

    ks = [r["k"] for r in results]

    # Per-method picks (for reference)

    inertias = np.array([r["inertia"] for r in results])
    d2       = np.diff(np.diff(inertias))
    elbow_k  = ks[1 + int(np.argmax(np.abs(d2)))]

    best_sil  = max(results, key=lambda r: r["silhouette"])

    # Tibshirani gap rule
    gap_k = ks[-1]
    for i in range(len(results) - 1):
        if results[i]["gap"] >= results[i + 1]["gap"] - results[i + 1]["gap_std"]:
            gap_k = ks[i]
            break

    best_ch   = max(results, key=lambda r: r["calinski_harabasz"])
    best_db   = min(results, key=lambda r: r["davies_bouldin"])
    best_stab = max(results, key=lambda r: r["stability"])

    # Composite score (weighted normalised)
    # Monotonic metrics (gap always ↑, CH always ↓, stability always ↓) get
    # low weights so they don't dominate.  DB is inverted (lower = better).

    sil_arr  = _norm01(np.array([r["silhouette"]        for r in results]))
    db_arr   = _norm01(np.array([r["davies_bouldin"]     for r in results]))
    ch_arr   = _norm01(np.array([r["calinski_harabasz"]  for r in results]))
    gap_arr  = _norm01(np.array([r["gap"]                for r in results]))

    # Elbow: normalise the absolute 2nd derivative of inertia
    d2_full = np.abs(np.concatenate([[0], d2, [0]]))   # pad to same length
    elbow_arr = _norm01(d2_full)

    composite = (
        0.30 * sil_arr
      + 0.25 * elbow_arr
      + 0.20 * (1 - db_arr)     # invert: lower DB = better
      + 0.15 * ch_arr
      + 0.10 * gap_arr
    )

    consensus_k = ks[int(np.argmax(composite))]

    # Store composite per row for JSON (useful for vis)
    for i, row in enumerate(results):
        row["composite"] = round(float(composite[i]), 4)

    print("\n  ── Per-method picks (informational) ───────────────────")
    print(f"  Elbow (2nd deriv):       k = {elbow_k}")
    print(f"  Silhouette (max):        k = {best_sil['k']}  (score={best_sil['silhouette']:.4f})")
    print(f"  Gap Statistic:           k = {gap_k}  (note: often k_max - monotone metric)")
    print(f"  Calinski-Harabasz (max): k = {best_ch['k']}  (CH={best_ch['calinski_harabasz']:.1f}  note: often k_min)")
    print(f"  Davies-Bouldin (min):    k = {best_db['k']}  (DB={best_db['davies_bouldin']:.4f})")
    print(f"  Stability (max Jaccard): k = {best_stab['k']}  (J={best_stab['stability']:.4f}  note: often k_min)")
    print(f"\n  ★ COMPOSITE (weighted normalised): k = {consensus_k}")
    print(f"    weights: Silhouettex0.30  Elbowx0.25  DB(inv)x0.20  CHx0.15  Gapx0.10")

    return {
        "results":       results,
        "k_range":       [ks[0], ks[-1]],
        "elbow_k":       elbow_k,
        "best_sil_k":    best_sil["k"],
        "best_sil_val":  best_sil["silhouette"],
        "gap_k":         gap_k,
        "best_ch_k":     best_ch["k"],
        "best_ch_val":   round(best_ch["calinski_harabasz"], 1),
        "best_db_k":     best_db["k"],
        "best_db_val":   round(best_db["davies_bouldin"], 4),
        "best_stab_k":   best_stab["k"],
        "best_stab_val": best_stab["stability"],
        "consensus_k":   consensus_k,
    }


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Find the optimal number of K-Means clusters (6 methods, composite score)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument("--embeddings", default=DEFAULT_EMB,
                    help="Path to embeddings .npy file")
    ap.add_argument("--output",     default=DEFAULT_KA,
                    help="Where to write k_metrics.json")
    ap.add_argument("--k-min",      type=int, default=4,
                    help="Smallest k to test (must be >= 2)")
    ap.add_argument("--k-max",      type=int, default=16,
                    help="Largest k to test")
    ap.add_argument("--seed",       type=int, default=SEED,
                    help="Random seed for reproducibility")
    a = ap.parse_args()

    if a.k_min < 2:
        ap.error("--k-min must be >= 2")
    if a.k_max <= a.k_min:
        ap.error("--k-max must be > --k-min")

    out_path = Path(a.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print(f"embeddings : {a.embeddings}")
    print(f"k range    : [{a.k_min}, {a.k_max}]")
    print(f"output     : {a.output}")
    print("=" * 60)

    print("\nLoading embeddings...")
    emb = load_embeddings(a.embeddings)
    print(f"  Shape: {emb.shape}")

    print(f"\nRunning analysis for k = {a.k_min} … {a.k_max} ...")
    ka = optimal_k_analysis(emb, k_range=range(a.k_min, a.k_max + 1), seed=a.seed)

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(ka, f, indent=2, ensure_ascii=False)
    print(f"Done  ({out_path.stat().st_size / 1024:.1f} KB)")

if __name__ == "__main__":
    main()
