"""
robustness_experiment.py
========================

Final robustness experiment — porównanie t-SNE, UMAP i PaCMAP
dla 10 niezależnych inicjalizacji każdej metody.

Dla każdego seeda i każdej metody:
1. Uruchamia projekcję z danym random_state
2. Oblicza wszystkie metryki jakości
3. Oblicza Choquet score

Raportuje:
- mean ± std Choquet score per metoda
- rankig per seed (która metoda wygrywa)
- p-value testu Wilcoxona między parami metod

Usage
-----
python robustness_experiment.py \
    --embeddings ../../wmii-data-collection/data/embeddings.npy \
    --input ../output/vis_methods.json \
    --seeds 10
"""

import argparse
import json
import warnings
from pathlib import Path

import numpy as np
from scipy.stats import wilcoxon

warnings.filterwarnings("ignore")

# ── Choquet (identyczny jak w choquet_composite.py) ──────────────────────────

SINGLETONS = {
    "knn":      0.17,
    "trust":    0.17,
    "nh":       0.16,
    "spearman": 0.15,
    "pearson":  0.15,
    "spde":     0.10,
    "stress":   0.10,
}

INTERACTIONS = {
    frozenset(["knn", "trust"]):        -0.04,
    frozenset(["spearman", "pearson"]): -0.03,
    frozenset(["knn", "nh"]):           +0.03,
    frozenset(["trust", "spde"]):       +0.04,
}


def fuzzy_measure(subset):
    subset = set(subset)
    if not subset:
        return 0.0
    val = sum(SINGLETONS[k] for k in subset)
    for pair, interaction in INTERACTIONS.items():
        if pair.issubset(subset):
            val += interaction
    return val


def choquet_integral(metrics):
    items  = sorted(metrics.items(), key=lambda x: x[1])
    values = [v for _, v in items]
    names  = [k for k, _ in items]
    total, prev = 0.0, 0.0
    for i in range(len(values)):
        total += (values[i] - prev) * fuzzy_measure(names[i:])
        prev = values[i]
    return total


def normalize_metrics(cm):
    return {
        "knn":      cm["knn_recall"],
        "trust":    cm["trustworthiness"],
        "nh":       cm["neighborhood_hit"],
        "spearman": (cm["spearman_r"] + 1) / 2,
        "pearson":  (cm["pearson_r"] + 1) / 2,
        "spde":     cm["distance_preservation"],
        "stress":   1 / (1 + cm["kruskal_stress"]),
    }


# ── Metryki jakości projekcji ─────────────────────────────────────────────────

def compute_metrics(emb_high, coords_2d, labels, sample_n=1500, seed=0):
    from scipy.stats import spearmanr
    from sklearn.metrics import pairwise_distances
    from sklearn.manifold import trustworthiness as sk_trust

    rng = np.random.default_rng(seed)
    idx = rng.choice(len(emb_high), min(sample_n, len(emb_high)), replace=False)
    es, cs, ls = emb_high[idx], coords_2d[idx], labels[idx]

    K = min(100, len(idx) - 1)
    Dc = pairwise_distances(es, metric="cosine")
    De = pairwise_distances(cs, metric="euclidean")
    iu = np.triu_indices(len(idx), k=1)
    dc, de = Dc[iu], De[iu]

    s01  = lambda a: (a - a.min()) / (a.max() - a.min() + 1e-12)
    dcs, des = s01(dc), s01(de)

    spde_val = float(1 - np.abs(dcs - des).mean())
    pearson  = float(np.corrcoef(dc, de)[0, 1])
    spear    = float(spearmanr(dc, de)[0])
    stress   = float(np.sqrt(np.sum((des - dcs)**2) / (np.sum(des**2) + 1e-12)))
    trust    = float(sk_trust(es, cs, n_neighbors=min(K, 50), metric="cosine"))

    knn_r = [
        len(set(np.argsort(Dc[i])[1:K+1]) & set(np.argsort(De[i])[1:K+1])) / K
        for i in range(len(idx))
    ]
    knn = float(np.mean(knn_r))
    nh  = float(np.mean([
        np.mean(ls[np.argsort(De[i])[1:11]] == ls[i])
        for i in range(len(idx))
    ]))

    return {
        "knn_recall":            round(knn,      4),
        "trustworthiness":       round(trust,    4),
        "neighborhood_hit":      round(nh,       4),
        "spearman_r":            round(spear,    4),
        "pearson_r":             round(pearson,  4),
        "distance_preservation": round(spde_val, 4),
        "kruskal_stress":        round(stress,   4),
    }


# ── DR runners ────────────────────────────────────────────────────────────────

def run_tsne(pca50, seed):
    from sklearn.manifold import TSNE
    try:
        proj = TSNE(
            n_components=2, random_state=seed, perplexity=35,
            metric="cosine", init="pca", learning_rate="auto",
            max_iter=1000,
        ).fit_transform(pca50)
    except TypeError:
        proj = TSNE(
            n_components=2, random_state=seed, perplexity=35,
            metric="cosine", n_iter=1000,
        ).fit_transform(pca50)
    return proj


def run_umap(emb, seed):
    import umap
    return umap.UMAP(
        n_components=2, n_neighbors=20, min_dist=0.1,
        metric="cosine", init="spectral", n_epochs=500,
        random_state=seed,
    ).fit_transform(emb)


def run_pacmap(emb, seed):
    import pacmap
    return pacmap.PaCMAP(
        n_components=2, n_neighbors=10, MN_ratio=0.5,
        FP_ratio=2.0, random_state=seed,
    ).fit_transform(emb)


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(
        description="Robustness experiment: t-SNE vs UMAP vs PaCMAP over N seeds"
    )
    ap.add_argument("--embeddings", required=True,
                    help="Path to embeddings.npy")
    ap.add_argument("--input",      required=True,
                    help="vis_methods.json (for cluster labels)")
    ap.add_argument("--seeds",      type=int, default=10,
                    help="Number of seeds (default: 10)")
    ap.add_argument("--sample",     type=int, default=1500,
                    help="Points used for metric computation (default: 1500)")
    ap.add_argument("--output",     default="robustness_results.json",
                    help="Where to write robustness_results.json (default: robustness_results.json)")
    args = ap.parse_args()

    # wczytaj dane
    from sklearn.preprocessing import normalize
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import MinMaxScaler

    emb = normalize(np.load(args.embeddings).astype(np.float32))
    with open(args.input, encoding="utf-8") as f:
        data = json.load(f)
    labels = np.array([p["cluster"] for p in data["points"]])

    print(f"Embeddings : {emb.shape}")
    print(f"Labels     : {labels.shape}  ({len(set(labels.tolist()))} clusters)")
    print(f"Seeds      : {args.seeds}")
    print()

    # PCA-50 dla t-SNE
    print("Computing PCA-50...")
    pca50 = PCA(n_components=50, random_state=42).fit_transform(emb)

    def scale(c):
        from sklearn.preprocessing import MinMaxScaler
        return MinMaxScaler(feature_range=(-1, 1)).fit_transform(c)

    seeds   = list(range(args.seeds))
    methods = ["t-SNE", "UMAP", "PaCMAP"]
    scores  = {m: [] for m in methods}
    metrics = {m: [] for m in methods}

    for seed in seeds:
        print(f"\n── seed {seed} " + "─" * 50)
        for method in methods:
            print(f"  [{method}]", end=" ", flush=True)
            if method == "t-SNE":
                proj = scale(run_tsne(pca50, seed))
            elif method == "UMAP":
                proj = scale(run_umap(emb, seed))
            else:
                proj = scale(run_pacmap(emb, seed))

            cm    = compute_metrics(emb, proj, labels,
                                    sample_n=args.sample, seed=seed)
            nm    = normalize_metrics(cm)
            score = choquet_integral(nm)
            scores[method].append(score)
            metrics[method].append(cm)
            print(f"Choquet={score:.4f}  kNN={cm['knn_recall']:.4f}  "
                  f"Trust={cm['trustworthiness']:.4f}")

        # ranking per seed
        seed_scores = {m: scores[m][-1] for m in methods}
        ranked = sorted(methods, key=lambda m: seed_scores[m], reverse=True)
        print(f"  Ranking: {' > '.join(f'{m}({seed_scores[m]:.4f})' for m in ranked)}")

    # ── Summary ───────────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("ROBUSTNESS EXPERIMENT — SUMMARY")
    print("=" * 70)
    print(f"  {'Method':<10} {'Mean':>8} {'Std':>8} {'Min':>8} {'Max':>8}  {'Wins':>5}")
    print("  " + "-" * 55)

    win_counts = {m: 0 for m in methods}
    for i in range(args.seeds):
        winner = max(methods, key=lambda m: scores[m][i])
        win_counts[winner] += 1

    arr = {}
    for m in methods:
        arr[m] = np.array(scores[m])
        print(f"  {m:<10} {arr[m].mean():>8.4f} {arr[m].std():>8.4f} "
              f"{arr[m].min():>8.4f} {arr[m].max():>8.4f}  "
              f"{win_counts[m]:>4}/{args.seeds}")

    # ── Testy statystyczne ────────────────────────────────────────────────────
    print()
    print("Wilcoxon signed-rank tests (paired, two-sided):")
    pairs = [("t-SNE", "UMAP"), ("t-SNE", "PaCMAP"), ("UMAP", "PaCMAP")]
    for a, b in pairs:
        try:
            stat, p = wilcoxon(arr[a], arr[b])
            sig = "**" if p < 0.01 else ("*" if p < 0.05 else "ns")
            print(f"  {a} vs {b}: p={p:.4f} {sig}")
        except Exception as e:
            print(f"  {a} vs {b}: {e}")

    # ── Per-seed tabela ───────────────────────────────────────────────────────
    print()
    print("Per-seed Choquet scores:")
    header = f"  {'Seed':>5}  " + "  ".join(f"{m:>10}" for m in methods) + "  Winner"
    print(header)
    print("  " + "-" * (len(header) - 2))
    for i, seed in enumerate(seeds):
        row = f"  {seed:>5}  " + "  ".join(f"{scores[m][i]:>10.4f}" for m in methods)
        winner = max(methods, key=lambda m: scores[m][i])
        print(row + f"  {winner}")

    # ── Zapis JSON ────────────────────────────────────────────────────────────
    out = {
        "n_seeds": args.seeds,
        "methods": {
            m: {
                "choquet_mean": round(float(arr[m].mean()), 4),
                "choquet_std":  round(float(arr[m].std()),  4),
                "choquet_min":  round(float(arr[m].min()),  4),
                "choquet_max":  round(float(arr[m].max()),  4),
                "wins":         win_counts[m],
                "per_seed": [
                    {"seed": seeds[i], "choquet": float(scores[m][i]),
                     **{k: float(v) for k, v in metrics[m][i].items()}}
                    for i in range(args.seeds)
                ],
            }
            for m in methods
        }
    }
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nResults saved → {out_path}")


if __name__ == "__main__":
    main()