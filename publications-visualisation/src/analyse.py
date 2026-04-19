"""
analyse.py
==========
Projects scientific-paper embeddings (768D) to 2D using eight
dimensionality-reduction methods and evaluates each with comprehensive
quality metrics.  Outputs a single JSON file consumed by vis_methods.html.

Typical workflow
----------------
  # Step 1 — decide on k  (run once, takes ~10 min)
  python find_optimal_k.py --embeddings data/embeddings.npy --k-min 5 --k-max 20

  # Step 2 — project & visualise  (uses the k chosen above)
  python analyse.py --embeddings data/embeddings.npy --clusters 9

  # Pass the k-analysis JSON so the HTML viewer shows the k-analysis tab:
  python analyse.py --embeddings data/embeddings.npy \
                    --clusters 9 \
                    --k-analysis wyniki/k_metrics.json

WHY ARE QUALITY SCORES "LOW"?
  kNN Recall of 0.15-0.57 is mathematically expected for 768D -> 2D.
  Johnson-Lindenstrauss requires ~1200D to preserve distances for 3440 points.
  Text embeddings have intrinsic dim ~20-100; projecting to 2D discards >90%
  of meaningful structure.  The scores are completely normal.

DR methods
----------
  UMAP       cosine 768D, state-of-the-art for text (BERTopic default)
  t-SNE      cosine + PCA-init (Kobak & Linderman 2021)
  PaCMAP     best local+global balance (ASA Chambers Award)
  Isomap     geodesic manifold distances
  Spectral   Graph Laplacian eigenvectors
  PCA        linear baseline
  PCA 8D+log 8-axis centroid pipeline (original approach)
  LDA        supervised; inter-cluster distances artificially inflated
"""

import argparse
import json
import time
import warnings
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import kendalltau, spearmanr
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.manifold import (
    Isomap,
    SpectralEmbedding,
    TSNE,
    trustworthiness as sklearn_trustworthiness,
)
from sklearn.metrics import pairwise_distances
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MinMaxScaler, normalize

from common import (
    COLORS, DEFAULT_EMB, DEFAULT_META, DEFAULT_OUT, DEFAULT_SCI,
    METRIC_N, N_CLUSTERS, SEED,
    load_embeddings, load_metadata, name_clusters,
)

warnings.filterwarnings("ignore")

try:
    import umap
    HAS_UMAP = True
except ImportError:
    HAS_UMAP = False

try:
    import pacmap
    HAS_PACMAP = True
except ImportError:
    HAS_PACMAP = False


# ── Clustering ────────────────────────────────────────────────────────────────

def cluster_data(emb, n=N_CLUSTERS):
    """Fit K-Means in 768D and return (labels, normalised centroids)."""
    print(f"\nK-Means  k={n}, 768D ...")
    km        = KMeans(n_clusters=n, random_state=SEED, n_init=10)
    labels    = km.fit_predict(emb)
    centroids = normalize(km.cluster_centers_)
    print(f"  Cluster sizes: {dict(Counter(labels.tolist()))}")
    return labels, centroids


# ── Quality metrics ───────────────────────────────────────────────────────────

def quality_metrics(emb, coords, labels, sample_n=METRIC_N, seed=SEED):
    """
    Compute a comprehensive set of 2D-projection quality metrics.

    Metrics
    -------
    kNN Recall@100    fraction of true 768D neighbours recovered in 2D
    Trustworthiness   fraction of 2D neighbours that are real 768D neighbours
    Neighbourhood Hit fraction of 2D neighbours sharing the same cluster label
    Spearman rho      rank correlation of pairwise distances
    Kendall tau       (subsampled) ordinal correlation
    Kruskal Stress    classic MDS stress (lower = better)
    SPDE              Scaled Pairwise Distance Error (global fidelity)
    Pearson r         linear correlation of pairwise distances
    Composite         0.4*kNN + 0.3*Trust + 0.2*Spearman + 0.1*(1-stress)
    """
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(emb), min(sample_n, len(emb)), replace=False)
    es, cs, ls = emb[idx], coords[idx], labels[idx]
    K = 100
    print(f"    {len(idx)} pts  ({len(idx) * (len(idx) - 1) // 2:,} pairs)")

    Dc = pairwise_distances(es, metric="cosine")
    De = pairwise_distances(cs, metric="euclidean")
    iu = np.triu_indices(len(idx), k=1)
    dc, de = Dc[iu], De[iu]

    s01  = lambda a: (a - a.min()) / (a.max() - a.min() + 1e-12)
    dcs, des = s01(dc), s01(de)

    err     = np.abs(dcs - des)
    spde    = float(1 - err.mean())
    pearson = float(np.corrcoef(dc, de)[0, 1])
    spear   = float(spearmanr(dc, de)[0])
    sub     = rng.choice(len(dc), min(50_000, len(dc)), replace=False)
    ktau    = float(kendalltau(dc[sub], de[sub])[0])
    stress  = float(np.sqrt(np.sum((des - dcs) ** 2) / (np.sum(des ** 2) + 1e-12)))
    trust   = float(sklearn_trustworthiness(es, cs, n_neighbors=K, metric="cosine"))

    knn_r = [
        len(set(np.argsort(Dc[i])[1:K + 1]) & set(np.argsort(De[i])[1:K + 1])) / K
        for i in range(len(idx))
    ]
    knn = float(np.mean(knn_r))

    nh = [
        float(np.mean(ls[np.argsort(De[i])[1:11]] == ls[i]))
        for i in range(len(idx))
    ]
    nbhit = float(np.mean(nh))

    comp = 0.4 * knn + 0.3 * trust + 0.2 * max(spear, 0) + 0.1 * (1 - min(stress / 2, 1))

    # Per-point SPDE on a 200-pt subsample (expensive)
    pp = []
    for i in idx[:200]:
        di_c = pairwise_distances(emb[[i]], emb,    metric="cosine")[0]
        di_e = pairwise_distances(coords[[i]], coords, metric="euclidean")[0]
        m    = np.ones(len(emb), dtype=bool)
        m[i] = False
        pp.append(float(1 - np.abs(s01(di_c[m]) - s01(di_e[m])).mean()))
    ppa = np.array(pp)

    return {
        "distance_preservation": round(spde, 4),
        "mae":                   round(float(err.mean()), 4),
        "max_error":             round(float(err.max()), 4),
        "pearson_r":             round(pearson, 4),
        "per_point_mean":        round(float(ppa.mean()), 4),
        "per_point_std":         round(float(ppa.std()), 4),
        "per_point_min":         round(float(ppa.min()), 4),
        "n_pairs":               int(len(idx) * (len(idx) - 1) // 2),
        "formula":               "SPDE = 1 - mean|scale(cos_768D) - scale(euc_2D)|",
        "spearman_r":            round(spear, 4),
        "kendall_tau":           round(ktau, 4),
        "kruskal_stress":        round(stress, 4),
        "trustworthiness":       round(trust, 4),
        "knn_recall":            round(knn, 4),
        "neighborhood_hit":      round(nbhit, 4),
        "composite_score":       round(float(comp), 4),
        "k_neighbors":           K,
    }


# ── DR helpers ────────────────────────────────────────────────────────────────

def _scale(c):
    return MinMaxScaler(feature_range=(-1, 1)).fit_transform(c)

def _pca50(e):
    return PCA(n_components=50, random_state=SEED).fit_transform(e)


# ── DR runners ────────────────────────────────────────────────────────────────

def run_umap_cosine(emb):
    """UMAP with cosine metric directly on 768D — state-of-the-art for text."""
    if not HAS_UMAP:
        raise ImportError("pip install umap-learn")
    t0 = time.time()
    c = umap.UMAP(
        n_components=2, n_neighbors=20, min_dist=0.1,
        metric="cosine", init="spectral", n_epochs=500,
        random_state=SEED,
    ).fit_transform(emb)
    return _scale(c), round(time.time() - t0, 2), {
        "note": "cosine 768D, spectral init, n_neighbors=20",
    }


def run_tsne(emb):
    """t-SNE with cosine metric + PCA init (Kobak & Linderman 2021)."""
    t0 = time.time()
    p  = _pca50(emb)
    try:
        c = TSNE(
            n_components=2, random_state=SEED, perplexity=35,
            metric="cosine", init="pca", learning_rate="auto",
            max_iter=1000,
        ).fit_transform(p)
    except TypeError:   # older sklearn
        c = TSNE(
            n_components=2, random_state=SEED, perplexity=35,
            metric="cosine", n_iter=1000,
        ).fit_transform(p)
    return _scale(c), round(time.time() - t0, 2), {
        "note": "cosine, PCA-init, perplexity=35",
        "warning": (
            "Inter-cluster distances artificially inflated by Student-t kernel. "
            "Cluster sizes equalized. Only within-cluster neighbourhoods are trustworthy."
        ),
    }


def run_pacmap(emb):
    """PaCMAP — best local+global balance (ASA Chambers Award)."""
    if not HAS_PACMAP:
        raise ImportError("pip install pacmap")
    t0 = time.time()
    c = pacmap.PaCMAP(
        n_components=2, n_neighbors=10, MN_ratio=0.5,
        FP_ratio=2.0, random_state=SEED,
    ).fit_transform(emb)
    return _scale(c), round(time.time() - t0, 2), {
        "note": "L2-normed input = cosine metric",
    }


def run_isomap(emb):
    """Isomap — geodesic manifold distances."""
    t0 = time.time()
    c  = Isomap(n_components=2, n_neighbors=15).fit_transform(_pca50(emb))
    return _scale(c), round(time.time() - t0, 2), {}


def run_spectral(emb):
    """Spectral Embedding — Graph Laplacian eigenvectors."""
    t0 = time.time()
    c  = SpectralEmbedding(
        n_components=2, random_state=SEED, n_neighbors=15,
    ).fit_transform(_pca50(emb))
    return _scale(c), round(time.time() - t0, 2), {}


def run_pca(emb):
    """PCA — linear baseline."""
    t0  = time.time()
    pca = PCA(n_components=2, random_state=SEED)
    c   = pca.fit_transform(emb)
    v   = float(pca.explained_variance_ratio_.sum() * 100)
    return _scale(c), round(time.time() - t0, 2), {
        "variance_explained": round(v, 2),
    }


def run_centroid(emb, centroids):
    """8-axis centroid pipeline: cosine -> log -> PCA (original approach)."""
    t0  = time.time()
    b   = cosine_similarity(emb, centroids)
    b   = MinMaxScaler().fit_transform(np.log(b - b.min() + 0.01))
    pca = PCA(n_components=2, random_state=SEED)
    c   = pca.fit_transform(b)
    v   = float(pca.explained_variance_ratio_.sum() * 100)
    return _scale(c), round(time.time() - t0, 2), {
        "variance_explained": round(v, 2),
        "note": "8-axis centroid log pipeline",
    }


def run_lda(emb, labels):
    """LDA — supervised; maximises between-cluster separation BY DESIGN."""
    t0 = time.time()
    c  = LinearDiscriminantAnalysis(n_components=2).fit_transform(emb, labels)
    return _scale(c), round(time.time() - t0, 2), {
        "warning": (
            "Supervised — maximises cluster separation by design. "
            "Inter-cluster distances are artificially inflated."
        ),
    }


# ── Method registry ───────────────────────────────────────────────────────────

METHODS_CONFIG = [
    ("umap",     "UMAP",       "UMAP cosine 768D (state-of-the-art)",               "non-linear"),
    ("tsne",     "t-SNE",      "t-SNE cosine PCA-init (best local fidelity)",        "non-linear"),
    ("pacmap",   "PaCMAP",     "PaCMAP (best local+global, ASA award)",              "non-linear"),
    ("isomap",   "Isomap",     "Isometric Mapping (geodesic distances)",             "non-linear"),
    ("spectral", "Spectral",   "Spectral Embedding (Laplacian Eigenmaps)",           "non-linear"),
    ("pca",      "PCA",        "Principal Component Analysis (linear baseline)",     "linear"),
    ("centroid", "PCA 8D+log", "8-Axis Centroid Pipeline (log transform)",           "linear"),
    ("lda",      "LDA",        "Linear Discriminant Analysis (supervised, biased)",  "linear"),
]


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(
        description="Project paper embeddings to 2D with 8 DR methods",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument("--embeddings",    default=DEFAULT_EMB,
                    help="Path to embeddings .npy")
    ap.add_argument("--metadata",      default=DEFAULT_META,
                    help="Path to paper metadata CSV")
    ap.add_argument("--scientists",    default=DEFAULT_SCI,
                    help="Path to scientists CSV (for author names)")
    ap.add_argument("--clusters",      type=int, default=N_CLUSTERS,
                    help="Number of K-Means clusters")
    ap.add_argument("--output",        default=DEFAULT_OUT,
                    help="Output JSON path for vis_methods.html")
    ap.add_argument("--k-analysis",    default=None,
                    help=(
                        "Optional path to k_metrics.json from "
                        "find_optimal_k.py — embeds it so the HTML viewer "
                        "shows the k-analysis tab"
                    ))
    ap.add_argument("--no-tsne",       action="store_true",
                    help="Skip t-SNE (saves ~5 min on large datasets)")
    ap.add_argument("--metric-sample", type=int, default=METRIC_N,
                    help="Points used when computing quality metrics")
    a = ap.parse_args()

    out_path = Path(a.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if not HAS_UMAP:
        print("\n  Warning: umap-learn not installed — pip install umap-learn")
    if not HAS_PACMAP:
        print("\n  Warning: pacmap not installed     — pip install pacmap")

    # ── Load ──────────────────────────────────────────────────────────────────
    print("Loading embeddings...")
    emb  = load_embeddings(a.embeddings)
    print(f"  Shape: {emb.shape}")
    meta = load_metadata(a.metadata, a.scientists)

    # ── Optional: ingest pre-computed k-analysis ──────────────────────────────
    ka = None
    if a.k_analysis:
        ka_path = Path(a.k_analysis)
        if ka_path.exists():
            with open(ka_path, encoding="utf-8") as f:
                ka = json.load(f)
            print(f"  [k-analysis] loaded  consensus k = {ka.get('consensus_k', '?')}")
        else:
            print(f"  [k-analysis] not found: {ka_path} — skipping")

    # ── Cluster ───────────────────────────────────────────────────────────────
    labels, centroids = cluster_data(emb, n=a.clusters)
    cnames            = name_clusters(meta, labels, a.clusters)

    # ── DR methods ────────────────────────────────────────────────────────────
    runners = {
        "umap":     lambda: run_umap_cosine(emb),
        "tsne":     lambda: run_tsne(emb),
        "pacmap":   lambda: run_pacmap(emb),
        "isomap":   lambda: run_isomap(emb),
        "spectral": lambda: run_spectral(emb),
        "pca":      lambda: run_pca(emb),
        "centroid": lambda: run_centroid(emb, centroids),
        "lda":      lambda: run_lda(emb, labels),
    }

    results = {}
    for key, short, desc, kind in METHODS_CONFIG:
        if key == "tsne" and a.no_tsne:
            print(f"\n[{short}] Skipped (--no-tsne)")
            continue
        print(f"\n[{short}] {desc}...")
        try:
            coords, elapsed, extra = runners[key]()
            print(f"  {elapsed}s — computing quality metrics...")
            cm = quality_metrics(emb, coords, labels, sample_n=a.metric_sample)
            print(f"  kNN={cm['knn_recall']:.4f}  Trust={cm['trustworthiness']:.4f}  "
                  f"Comp={cm['composite_score']:.4f}  SPDE={cm['distance_preservation']:.4f}")
            results[key] = {
                "key": key, "short": short, "long": desc, "kind": kind,
                "elapsed_s": elapsed, "coords": coords.tolist(),
                "custom_metric": cm, **extra,
            }
        except ImportError as e:
            print(f"  SKIPPED — {e}")
        except Exception as e:
            print(f"  ERROR   — {e}")

    method_order = [m[0] for m in METHODS_CONFIG if m[0] in results]

    # ── Wheel axes ────────────────────────────────────────────────────────────
    print("\nBuilding wheel axes...")
    b  = cosine_similarity(emb, centroids)
    bs = MinMaxScaler().fit_transform(np.log(b - b.min() + 0.01))
    c2 = PCA(n_components=2, random_state=SEED).fit_transform(centroids)
    wheel_angles = np.arctan2(c2[:, 1], c2[:, 0]).tolist()

    # ── Point records ─────────────────────────────────────────────────────────
    print("Building point records...")
    pts = []
    for i in range(len(labels)):
        p = {"cluster": int(labels[i])}
        if meta is not None and i < len(meta):
            r  = meta.iloc[i]
            yr = r.get("publication_year", None)
            p.update({
                "title":  str(r.get("title",             "")) or None,
                "author": str(r.get("full_name",         "")) or None,
                "orcid":  str(r.get("main_author_orcid", "")) or None,
                "year":   int(yr) if pd.notna(yr) else None,
                "id":     str(r.get("openalex_id", i)),
            })
        else:
            p.update({"title": None, "author": None, "orcid": None,
                      "year": None, "id": str(i)})
        for k in results:
            c = results[k]["coords"]
            p[f"{k}_x"] = round(c[i][0], 5)
            p[f"{k}_y"] = round(c[i][1], 5)
        p["axes"] = [round(float(bs[i, c]), 4) for c in range(a.clusters)]
        pts.append(p)

    # ── Assemble output JSON ──────────────────────────────────────────────────
    def _year(col, fn):
        return int(fn(meta[col])) if (meta is not None and col in meta.columns) else None

    out = {
        "n_points":      len(pts),
        "n_clusters":    a.clusters,
        "colors":        COLORS[:a.clusters],
        "cluster_names": [cnames[i] for i in range(a.clusters)],
        "has_metadata":  meta is not None,
        "year_range":    [_year("publication_year", min) or 1970,
                          _year("publication_year", max) or 2025],
        "methods":       {k: {kk: vv for kk, vv in v.items() if kk != "coords"}
                          for k, v in results.items()},
        "method_order":  method_order,
        "wheel_angles":  wheel_angles,
        "points":        pts,
    }
    if ka:
        out["k_analysis"] = ka

    print(f"\nWriting -> {out_path}")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, separators=(",", ":"), ensure_ascii=False)
    print(f"Done  ({out_path.stat().st_size / 1e6:.2f} MB)")

    # ── Terminal summary ──────────────────────────────────────────────────────
    print("\n" + "=" * 88)
    print("  QUALITY METRICS — kNN Recall 0.15-0.57 is NORMAL for 768D -> 2D")
    print("=" * 88)
    print(f"  {'Method':<14} {'kNN':>7} {'Trust':>7} {'Comp':>7} "
          f"{'NbHit':>7} {'Spear':>7} {'SPDE':>7} {'Stress':>8}")
    print(f"  {'-'*14} {'-'*7} {'-'*7} {'-'*7} {'-'*7} {'-'*7} {'-'*7} {'-'*8}")
    for k in method_order:
        m  = out["methods"][k]
        cm = m.get("custom_metric", {})
        w  = "  !" if m.get("warning") else ""
        print(f"  {m['short']:<14} "
              f"{cm.get('knn_recall',0):>7.4f} "
              f"{cm.get('trustworthiness',0):>7.4f} "
              f"{cm.get('composite_score',0):>7.4f} "
              f"{cm.get('neighborhood_hit',0):>7.4f} "
              f"{cm.get('spearman_r',0):>7.4f} "
              f"{cm.get('distance_preservation',0):>7.4f} "
              f"{cm.get('kruskal_stress',0):>8.4f}{w}")

    print("\n-- Ranking by Composite Score --")
    ranked = sorted(
        [(k, out["methods"][k]) for k in method_order
         if "custom_metric" in out["methods"][k]],
        key=lambda x: x[1]["custom_metric"].get("composite_score", 0),
        reverse=True,
    )
    for rank, (k, m) in enumerate(ranked, 1):
        cm = m["custom_metric"]
        w  = "  <- supervised bias" if m.get("warning") else ""
        print(f"  {rank}. {m['short']:<14}  "
              f"Comp={cm['composite_score']:.4f}  "
              f"kNN={cm['knn_recall']:.4f}  "
              f"Trust={cm['trustworthiness']:.4f}{w}")


if __name__ == "__main__":
    main()