"""
find_optimal_k.py
=================
Analyses the optimal number of K-Means clusters for a set of embeddings
using six complementary methods and writes the results to JSON.

Methods
-------
  1. Elbow / Inertia        — kink in SSE curve (2nd-derivative)
  2. Silhouette Score       — mean cohesion vs. separation  ↑ higher = better
  3. Gap Statistic          — vs. random uniform reference  ↑ higher = better
  4. Calinski-Harabasz (CH) — between/within variance ratio ↑ higher = better
  5. Davies-Bouldin (DB)    — average cluster similarity    ↓ lower  = better
  6. Stability              — bootstrap Jaccard consistency ↑ higher = better

A majority-vote consensus across all six methods is also reported.

Output
------
  k_metrics.json  — standalone file readable by vis_methods.html

Usage
-----
  python find_optimal_k.py --embeddings data/embeddings.npy
  python find_optimal_k.py --embeddings data/embeddings.npy --k-min 5 --k-max 20
  python find_optimal_k.py --help
"""

import argparse
import json
import warnings
from collections import Counter
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


# ── Internal helpers ──────────────────────────────────────────────────────────

def _gap_statistic(
    pca50: np.ndarray,
    km: KMeans,
    k: int,
    n_refs: int = 8,
    seed: int = SEED,
) -> tuple[float, float]:
    """
    Gap Statistic (Tibshirani et al. 2001).

    Compares log(inertia) of the real clustering against the expected
    log(inertia) of k-means run on n_refs uniform random datasets drawn
    from the bounding box of the PCA-50 space.

    Returns
    -------
    gap     : float  — gap value (higher → real clusters beat random)
    gap_std : float  — standard error used for the Tibshirani stopping rule
    """
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
    """
    Bootstrap stability via Jaccard co-clustering agreement.

    Fits KMeans on `n_runs` random 80 % subsamples, then for every pair of
    runs computes the Jaccard index on shared points:

        J = |same cluster in both| / |same cluster in at least one|

    A high average J means the solution is reproducible regardless of which
    points are included — i.e. the cluster structure is stable.
    """
    rng   = np.random.default_rng(seed)
    n     = len(emb)
    sub_n = int(n * subsample)

    all_labels  : list[np.ndarray] = []
    all_indices : list[np.ndarray] = []

    for _ in range(n_runs):
        idx = rng.choice(n, sub_n, replace=False)
        km  = KMeans(n_clusters=k, random_state=int(rng.integers(1_000_000)), n_init=5)
        lab = km.fit_predict(emb[idx])
        all_labels.append(lab)
        all_indices.append(np.sort(idx))   # keep sorted for searchsorted

    jaccard_scores: list[float] = []
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


# ── Main analysis ─────────────────────────────────────────────────────────────

def optimal_k_analysis(
    emb: np.ndarray,
    k_range: range = range(5, 21),
    seed: int = SEED,
) -> dict:
    """
    Run all six cluster-count methods for every k in k_range.

    Returns a dict with:
      results       — list of per-k metric rows
      elbow_k       — k from elbow method
      best_sil_k/v  — k + value from silhouette
      gap_k         — k from Tibshirani Gap rule
      best_ch_k/v   — k + value from Calinski-Harabasz
      best_db_k/v   — k + value from Davies-Bouldin
      best_stab_k/v — k + value from stability
      consensus_k   — majority vote across all six
      votes         — dict of {k: vote_count}
    """
    rng = np.random.default_rng(seed)

    # Subsample for silhouette (expensive at full N)
    sidx    = rng.choice(len(emb), min(2000, len(emb)), replace=False)
    emb_sub = emb[sidx]

    # PCA-50 used only for Gap reference generation (faster than 768D uniform box)
    print("  Computing PCA-50 for Gap reference space...")
    pca50 = PCA(n_components=min(50, emb.shape[1]), random_state=seed).fit_transform(emb)

    results: list[dict] = []

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
            "k":                  k,
            "inertia":            round(inertia, 1),
            "silhouette":         round(sil, 4),
            "gap":                gap,
            "gap_std":            gstd,
            "calinski_harabasz":  round(ch, 1),
            "davies_bouldin":     round(db, 4),
            "stability":          stability,
        }
        results.append(row)
        print(f"  {k:>3}  {inertia:>10.1f}  {sil:>10.4f}  {gap:>7.4f}  "
              f"{gstd:>6.4f}  {ch:>9.1f}  {db:>7.4f}  {stability:>9.4f}")

    # ── Derive recommended k per method ──────────────────────────────────────
    ks       = list(k_range)
    inertias = [r["inertia"] for r in results]

    # 1. Elbow — largest absolute 2nd derivative of inertia
    d2      = np.diff(np.diff(inertias))
    elbow_k = ks[1 + int(np.argmax(np.abs(d2)))]

    # 2. Silhouette — highest score
    best_sil = max(results, key=lambda r: r["silhouette"])

    # 3. Gap — Tibshirani rule: first k where gap(k) >= gap(k+1) - s(k+1)
    gap_k = ks[-1]   # conservative fallback
    for i in range(len(results) - 1):
        if results[i]["gap"] >= results[i + 1]["gap"] - results[i + 1]["gap_std"]:
            gap_k = ks[i]
            break

    # 4. Calinski-Harabasz — highest
    best_ch = max(results, key=lambda r: r["calinski_harabasz"])

    # 5. Davies-Bouldin — lowest
    best_db = min(results, key=lambda r: r["davies_bouldin"])

    # 6. Stability — highest Jaccard
    best_stab = max(results, key=lambda r: r["stability"])

    # Consensus — majority vote
    votes       = Counter([elbow_k, best_sil["k"], gap_k, best_ch["k"], best_db["k"], best_stab["k"]])
    consensus_k = votes.most_common(1)[0][0]

    print("\n  ── Recommendations ──────────────────────────────────")
    print(f"  Elbow (2nd deriv):       k = {elbow_k}")
    print(f"  Silhouette (max):        k = {best_sil['k']}  (score={best_sil['silhouette']:.4f})")
    print(f"  Gap Statistic:           k = {gap_k}")
    print(f"  Calinski-Harabasz (max): k = {best_ch['k']}  (CH={best_ch['calinski_harabasz']:.1f})")
    print(f"  Davies-Bouldin (min):    k = {best_db['k']}  (DB={best_db['davies_bouldin']:.4f})")
    print(f"  Stability (max Jaccard): k = {best_stab['k']}  (J={best_stab['stability']:.4f})")
    print(f"  ★ CONSENSUS (majority):  k = {consensus_k}  votes={dict(votes)}")

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
        "votes":         dict(votes),
    }


# ── CLI ───────────────────────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser(
        description="Find the optimal number of K-Means clusters (6 methods)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument("--embeddings", default=DEFAULT_EMB,
                    help="Path to embeddings .npy file")
    ap.add_argument("--output",     default=DEFAULT_KA,
                    help="Where to write k_metrics.json")
    ap.add_argument("--k-min",      type=int, default=6,
                    help="Smallest k to test (must be >= 2)")
    ap.add_argument("--k-max",      type=int, default=20,
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
    print("  find_optimal_k.py")
    print(f"  embeddings : {a.embeddings}")
    print(f"  k range    : [{a.k_min}, {a.k_max}]")
    print(f"  output     : {a.output}")
    print("=" * 60)

    print("\nLoading embeddings...")
    emb = load_embeddings(a.embeddings)
    print(f"  Shape: {emb.shape}")

    print(f"\nRunning analysis for k = {a.k_min} … {a.k_max}  (this may take a few minutes)...")
    ka = optimal_k_analysis(emb, k_range=range(a.k_min, a.k_max + 1), seed=a.seed)

    print(f"\nWriting -> {out_path}")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(ka, f, indent=2, ensure_ascii=False)
    print(f"Done  ({out_path.stat().st_size / 1024:.1f} KB)")
    print(f"\n  Next step: python analyse.py --clusters {ka['consensus_k']}")


if __name__ == "__main__":
    main()