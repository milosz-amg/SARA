"""
optimize_weights.py
===================
Finds the optimal weights for the DR quality composite score by maximising
the average Spearman rank correlation between the weighted composite and
each individual quality metric.

Idea
----
Each individual metric already provides a ranking of DR methods.
The "best" composite is the one whose ranking agrees most with the
consensus of ALL individual metrics simultaneously — i.e. it minimises
disagreement across metrics rather than favouring any single one.

This is a data-driven, principled approach that avoids arbitrary weight
assignment.  The resulting weights can be reported in a thesis as:
  "Weights were optimised to maximise the mean Spearman rank correlation
   between the composite score and each individual quality metric
   (Kendall, 1938; rank aggregation literature)."

Usage
-----
  python optimize_weights.py                         # uses output/vis_methods.json
  python optimize_weights.py --input path/to/vis_methods.json
  python optimize_weights.py --plot                  # show weight sensitivity plot

Output
------
  Prints recommended weights and their justification.
  Optionally saves a sensitivity plot to output/weight_sensitivity.png
"""

import argparse
import json
import warnings
from itertools import product
from pathlib import Path

import numpy as np
from scipy.optimize import minimize
from scipy.stats import spearmanr

warnings.filterwarnings("ignore")

HERE    = Path(__file__).parent
OUT_DIR = HERE.parent / "output"

# ── Metrics and their direction (higher = better after normalisation) ─────────
METRICS = {
    "knn_recall":            +1,   # higher = better
    "trustworthiness":       +1,
    "neighborhood_hit":      +1,
    "spearman_r":            +1,
    "distance_preservation": +1,   # SPDE
    "kruskal_stress":        -1,   # lower = better  →  negate
    "pearson_r":             +1,
}

# Metrics to EXCLUDE from composite (they measure different things or are
# redundant / noisy at this sample size)
EXCLUDE = {"kendall_tau", "mae", "max_error",
           "per_point_mean", "per_point_std", "per_point_min"}

COMPOSITE_METRICS = {k: v for k, v in METRICS.items() if k not in EXCLUDE}


def load_scores(json_path: Path) -> tuple[list[str], dict[str, np.ndarray]]:
    """
    Returns:
        methods  : list of DR method keys in order
        scores   : dict metric_name -> np.ndarray of raw values (one per method)
    """
    with open(json_path, encoding="utf-8") as f:
        data = json.load(f)

    methods = data["method_order"]
    scores  = {m: [] for m in COMPOSITE_METRICS}

    for key in methods:
        cm = data["methods"][key].get("custom_metric", {})
        for metric, direction in COMPOSITE_METRICS.items():
            val = cm.get(metric, None)
            scores[metric].append(float(val) if val is not None else np.nan)

    return methods, {m: np.array(v) for m, v in scores.items()}


def normalise(arr: np.ndarray, direction: int) -> np.ndarray:
    """Min-max normalise, flip if lower=better."""
    a = arr.copy()
    mn, mx = np.nanmin(a), np.nanmax(a)
    if mx - mn < 1e-12:
        return np.zeros_like(a)
    a = (a - mn) / (mx - mn)
    return a if direction == +1 else 1 - a


def composite(weights: np.ndarray, norm_scores: dict) -> np.ndarray:
    """Weighted sum of normalised scores."""
    keys = list(norm_scores.keys())
    return sum(weights[i] * norm_scores[keys[i]] for i in range(len(keys)))


def objective(weights: np.ndarray, norm_scores: dict) -> float:
    """
    Negative mean Spearman correlation between composite and each individual
    metric.  We negate because scipy.optimize minimises.
    """
    w = np.abs(weights)
    w = w / w.sum()   # ensure sum to 1

    comp = composite(w, norm_scores)
    corrs = []
    for arr in norm_scores.values():
        if np.std(arr) < 1e-10 or np.std(comp) < 1e-10:
            continue
        r, _ = spearmanr(comp, arr)
        corrs.append(r)

    return -float(np.mean(corrs)) if corrs else 0.0


def grid_search(norm_scores: dict, n_grid: int = 20) -> np.ndarray:
    """
    Coarse grid search over the weight simplex as initialisation for
    the continuous optimiser.  Tries n_grid^(k-1) combinations.
    """
    k = len(norm_scores)
    best_obj, best_w = np.inf, None

    # For k metrics use random sampling of the simplex (faster than full grid)
    rng = np.random.default_rng(42)
    n_samples = min(5000, n_grid ** (k - 1))

    for _ in range(n_samples):
        w = rng.dirichlet(np.ones(k))
        obj = objective(w, norm_scores)
        if obj < best_obj:
            best_obj = obj
            best_w   = w.copy()

    return best_w


def optimise_weights(norm_scores: dict) -> tuple[np.ndarray, float]:
    """
    Find weights that maximise mean Spearman correlation between composite
    and each individual metric.  Uses grid search + L-BFGS-B refinement.
    """
    k = len(norm_scores)
    print(f"  Optimising {k} weights via rank correlation maximisation...")

    # 1. grid search for good starting point
    w0 = grid_search(norm_scores, n_grid=20)
    print(f"  Grid search best obj: {-objective(w0, norm_scores):.4f}")

    # 2. continuous optimisation (constrained: weights sum to 1, all >= 0)
    constraints = {"type": "eq", "fun": lambda w: np.sum(np.abs(w)) - 1}
    bounds      = [(0, 1)] * k

    result = minimize(
        objective,
        x0     = w0,
        args   = (norm_scores,),
        method = "SLSQP",
        bounds = bounds,
        constraints = constraints,
        options = {"maxiter": 1000, "ftol": 1e-9},
    )

    w_opt = np.abs(result.x)
    w_opt = w_opt / w_opt.sum()
    mean_corr = -result.fun

    return w_opt, mean_corr


def sensitivity_analysis(norm_scores: dict, w_opt: np.ndarray,
                         n_perturb: int = 2000) -> dict:
    """
    Perturb each weight ±ε and measure impact on objective.
    Returns dict metric -> sensitivity score (higher = more important).
    """
    keys = list(norm_scores.keys())
    base = -objective(w_opt, norm_scores)
    sensitivities = {}

    rng = np.random.default_rng(42)
    impacts = {k: [] for k in keys}

    for _ in range(n_perturb):
        i = rng.integers(len(keys))
        w_perturbed = w_opt.copy()
        delta = rng.uniform(-0.1, 0.1)
        w_perturbed[i] += delta
        w_perturbed = np.clip(w_perturbed, 0, 1)
        if w_perturbed.sum() > 1e-10:
            w_perturbed /= w_perturbed.sum()
        new_obj = -objective(w_perturbed, norm_scores)
        impacts[keys[i]].append(abs(new_obj - base))

    for k in keys:
        sensitivities[k] = float(np.mean(impacts[k])) if impacts[k] else 0.0

    return sensitivities


def print_results(methods, raw_scores, norm_scores, w_opt, mean_corr):
    keys = list(norm_scores.keys())

    print("\n" + "=" * 60)
    print("  OPTIMAL WEIGHTS — DR Composite Score")
    print("=" * 60)
    print(f"  Mean Spearman correlation with individual metrics: {mean_corr:.4f}")
    print()
    print(f"  {'Metric':<28} {'Weight':>8}  {'Raw range'}")
    print(f"  {'-'*28} {'-'*8}  {'-'*20}")
    for i, metric in enumerate(keys):
        arr = raw_scores[metric]
        direction = COMPOSITE_METRICS[metric]
        arrow = "↑" if direction == +1 else "↓"
        print(f"  {metric:<28} {w_opt[i]:>8.4f}  "
              f"[{np.nanmin(arr):.4f} – {np.nanmax(arr):.4f}]  {arrow}")

    print()
    print("  Composite formula:")
    terms = " + ".join(
        f"{w_opt[i]:.2f}×{k}" for i, k in enumerate(keys) if w_opt[i] > 0.005
    )
    print(f"  comp = {terms}")

    # Ranking of methods
    comp = composite(w_opt, norm_scores)
    ranked = sorted(zip(comp, methods), reverse=True)
    print()
    print(f"  {'Rank':<6} {'Method':<14} {'Score':>8}")
    print(f"  {'-'*6} {'-'*14} {'-'*8}")
    for rank, (score, method) in enumerate(ranked, 1):
        print(f"  {rank:<6} {method:<14} {score:>8.4f}")

    # Compare with equal weights
    eq_w = np.ones(len(keys)) / len(keys)
    comp_eq = composite(eq_w, norm_scores)
    ranked_eq = [m for _, m in sorted(zip(comp_eq, methods), reverse=True)]
    ranked_opt = [m for _, m in ranked]

    if ranked_opt == ranked_eq:
        print("\n  ✓ Optimal weights produce the same ranking as equal weights")
    else:
        print("\n  Rankings differ from equal weights:")
        print(f"    Equal weights: {' > '.join(ranked_eq)}")
        print(f"    Optimal:       {' > '.join(ranked_opt)}")

    print()
    print("  Spearman correlations between composite and each metric:")
    for metric, arr in norm_scores.items():
        r, _ = spearmanr(comp, arr)
        print(f"    {metric:<28} r = {r:.4f}")

    print()
    print("  Thesis citation:")
    print("    \"Weights were determined empirically by maximising the mean")
    print("    Spearman rank correlation between the composite score and each")
    print("    individual quality metric across all evaluated DR methods.")
    print("    This ensures the composite reflects the consensus of all metrics")
    print("    rather than favouring any single measure.\"")
    print("=" * 60)

    return ranked_opt


def plot_sensitivity(norm_scores, w_opt, out_path: Path):
    try:
        import matplotlib.pyplot as plt
        import matplotlib
        matplotlib.use("Agg")
    except ImportError:
        print("  matplotlib not installed — skipping plot")
        return

    keys   = list(norm_scores.keys())
    labels = [k.replace("_", " ") for k in keys]
    base   = -objective(w_opt, norm_scores)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # ── Left: optimal weights bar chart ──────────────────────────────────────
    ax = axes[0]
    colors = ["#5b8def" if w >= 0.1 else "#aaaaaa" for w in w_opt]
    bars = ax.barh(labels, w_opt, color=colors, edgecolor="white", height=0.6)
    ax.set_xlabel("Optimal weight", fontsize=10)
    ax.set_title("Optimal weights maximise mean Spearman r", fontsize=10)
    ax.axvline(1 / len(keys), color="red", lw=1, linestyle="--",
               label=f"Equal weight (1/{len(keys)})")
    ax.legend(fontsize=8)
    for bar, w in zip(bars, w_opt):
        ax.text(bar.get_width() + 0.003, bar.get_y() + bar.get_height() / 2,
                f"{w:.3f}", va="center", fontsize=8)
    ax.set_xlim(0, max(w_opt) * 1.25)
    ax.invert_yaxis()

    # ── Right: impact of weight change on mean Spearman r ────────────────────
    ax2 = axes[1]
    deltas = np.linspace(-0.25, 0.25, 50)
    for i, (metric, label) in enumerate(zip(keys, labels)):
        impacts = []
        for d in deltas:
            w = w_opt.copy()
            w[i] = max(0, w[i] + d)
            if w.sum() > 1e-10:
                w /= w.sum()
            impacts.append(-objective(w, norm_scores))
        ax2.plot(deltas, impacts, lw=1.5, label=label, alpha=0.8)

    ax2.axvline(0, color="black", lw=1, linestyle="--", alpha=0.4)
    ax2.axhline(base, color="black", lw=0.8, linestyle=":", alpha=0.4,
                label=f"Optimal ({base:.3f})")
    ax2.set_xlabel("Δweight (perturbation from optimal)", fontsize=10)
    ax2.set_ylabel("Mean Spearman r across all metrics", fontsize=10)
    ax2.set_title("Sensitivity: effect of perturbing each weight", fontsize=10)
    ax2.legend(fontsize=7, ncol=2)

    plt.suptitle("DR composite score — weight analysis", fontsize=11, y=1.02)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"  Plot saved → {out_path}")


def main():
    ap = argparse.ArgumentParser(
        description="Find optimal DR composite score weights via rank correlation",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument("--input", default=OUT_DIR / "vis_methods.json",
                    help="Path to vis_methods.json")
    ap.add_argument("--plot", action="store_true",
                    help="Save weight sensitivity plot")
    a = ap.parse_args()

    json_path = Path(a.input)
    if not json_path.exists():
        print(f"ERROR: {json_path} not found")
        print("Run analyse.py first to generate vis_methods.json")
        return

    print("=" * 60)
    print("  optimize_weights.py")
    print(f"  Input: {json_path}")
    print("=" * 60)

    print("\nLoading scores...")
    methods, raw_scores = load_scores(json_path)
    print(f"  {len(methods)} DR methods: {methods}")
    print(f"  {len(raw_scores)} metrics: {list(raw_scores.keys())}")

    # Normalise
    norm_scores = {
        m: normalise(raw_scores[m], COMPOSITE_METRICS[m])
        for m in raw_scores
    }

    # Optimise
    w_opt, mean_corr = optimise_weights(norm_scores)

    # Print results
    print_results(methods, raw_scores, norm_scores, w_opt, mean_corr)

    # Sensitivity plot
    if a.plot:
        plot_sensitivity(norm_scores, w_opt,
                         json_path.parent / "weight_sensitivity.png")

    # Save weights to JSON for use in analyse.py
    out = {
        "weights": {
            m: round(float(w_opt[i]), 4)
            for i, m in enumerate(norm_scores.keys())
        },
        "mean_spearman_r": round(mean_corr, 4),
        "method": "max mean Spearman rank correlation across all individual metrics",
    }
    out_path = json_path.parent / "optimal_weights.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\n  Weights saved → {out_path}")
    print(f"  Pass to analyse.py with: --weights {out_path}")


if __name__ == "__main__":
    main()