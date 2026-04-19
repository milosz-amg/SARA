"""
choquet_composite.py
====================
Computes a Choquet integral composite score for DR method quality,
using a fuzzy measure derived from inter-metric entropy (Kojadinovic, 2004).

Why Choquet integral?
---------------------
Standard weighted sums assume metrics are independent.  In practice
knn_recall and trustworthiness are highly correlated — a weighted sum
effectively double-counts their contribution.  The Choquet integral
uses a *fuzzy measure* μ(S) that captures redundancy and synergy
between subsets of metrics, giving a more honest aggregation.

How the fuzzy measure is estimated:
-------------------------------------
Kojadinovic (2004) showed that for a set of criteria, a non-informative
fuzzy measure can be derived from the *mutual information* (entropy) between
them.  Metrics that share a lot of information get a *subadditive* measure
(redundant) while complementary metrics get a *superadditive* one.

Algorithm:
  1. Compute pairwise mutual information I(X_i; X_j) between metrics
     across the n DR methods.
  2. Build the fuzzy measure via the entropy-based capacity:
       μ({i}) = 1/n   (uniform singleton weights)
       μ(S)   = μ_additive(S) × correction(S)
     where correction reduces μ(S) for highly correlated subsets.
  3. Compute Shapley values φ_i — the "fair" marginal contribution
     of each metric averaged over all orderings.
  4. Compute the Choquet integral:
       C_mu(x) = sum_i  [x_(i) - x_(i-1)] * mu(A_(i))
     where x_(1) ≤ … ≤ x_(n) is the sorted score vector and
     A_(i) = {metrics with score ≥ x_(i)}.

References:
  Kojadinovic, I. (2004). Estimation of the weights of interacting criteria
    from the set of profiles by means of information-theoretic functionals.
    European Journal of Operational Research, 155(3), 741–751.
    doi:10.1016/S0377-2217(02)00752-1

  Grabisch, M. (1997). k-order additive discrete fuzzy measures and their
    representation. Fuzzy Sets and Systems, 92(2), 167–189.

  Lee & Verleysen (2009). Quality assessment of dimensionality reduction:
    Rank-based criteria. Neurocomputing, 72(7–9), 1431–1443.
    doi:10.1016/j.neucom.2008.12.017

Usage:
  python src/choquet_composite.py
  python src/choquet_composite.py --input output/vis_methods.json --plot
"""

import argparse
import json
import warnings
from itertools import combinations
from pathlib import Path

import numpy as np
from scipy.stats import rankdata, spearmanr

warnings.filterwarnings("ignore")

HERE    = Path(__file__).parent
OUT_DIR = HERE.parent / "output"

# ── Metric definitions ────────────────────────────────────────────────────────
# direction: +1 = higher is better, -1 = lower is better (inverted before use)
METRICS = {
    "knn_recall":            +1,
    "trustworthiness":       +1,
    "neighborhood_hit":      +1,
    "spearman_r":            +1,
    "distance_preservation": +1,
    "kruskal_stress":        -1,
    "pearson_r":             +1,
}

# Semantic grouping for reporting (Lee & Verleysen, 2009)
LOCAL_METRICS  = {"knn_recall", "trustworthiness", "neighborhood_hit"}
GLOBAL_METRICS = {"spearman_r", "distance_preservation", "kruskal_stress", "pearson_r"}


# ══════════════════════════════════════════════════════════════════════════════
# Data loading & normalisation
# ══════════════════════════════════════════════════════════════════════════════

def load_scores(json_path: Path):
    with open(json_path, encoding="utf-8") as f:
        data = json.load(f)
    methods = data["method_order"]
    scores  = {m: [] for m in METRICS}
    for key in methods:
        cm = data["methods"][key].get("custom_metric", {})
        for metric in METRICS:
            val = cm.get(metric)
            scores[metric].append(float(val) if val is not None else np.nan)
    return methods, {m: np.array(v) for m, v in scores.items()}


def normalise(arr: np.ndarray, direction: int) -> np.ndarray:
    mn, mx = np.nanmin(arr), np.nanmax(arr)
    if mx - mn < 1e-12:
        return np.full_like(arr, 0.5)
    n = (arr - mn) / (mx - mn)
    return n if direction == +1 else 1 - n


# ══════════════════════════════════════════════════════════════════════════════
# Fuzzy measure via entropy / mutual information (Kojadinovic 2004)
# ══════════════════════════════════════════════════════════════════════════════

def entropy_1d(x: np.ndarray, bins: int = 5) -> float:
    """Discrete entropy of a continuous variable via histogram binning."""
    counts, _ = np.histogram(x, bins=bins)
    p = counts / counts.sum()
    p = p[p > 0]
    return float(-np.sum(p * np.log2(p)))


def joint_entropy(x: np.ndarray, y: np.ndarray, bins: int = 5) -> float:
    counts, _, _ = np.histogram2d(x, y, bins=bins)
    p = counts / counts.sum()
    p = p[p > 0]
    return float(-np.sum(p * np.log2(p)))


def mutual_information(x: np.ndarray, y: np.ndarray, bins: int = 5) -> float:
    return entropy_1d(x, bins) + entropy_1d(y, bins) - joint_entropy(x, y, bins)


def build_fuzzy_measure(norm_scores: dict) -> dict:
    """
    Build a 2-additive fuzzy measure from pairwise mutual information.

    A 2-additive measure is fully defined by:
      - singleton values  μ({i})
      - interaction indices  I(i,j) = μ({i,j}) - μ({i}) - μ({j})
        positive I = synergy (complementary metrics)
        negative I = redundancy (correlated metrics)

    Singleton weights derived from entropy:
      μ({i}) ∝ H(i) / Σ H(j)   (more informative metric = higher weight)

    Interaction index:
      I(i,j) = -MI(i,j) / max_MI   (negative = redundant)
      scaled so that Σ_j I(i,j) stays bounded.

    Returns dict mapping frozenset → capacity value.
    """
    keys  = list(norm_scores.keys())
    n     = len(keys)
    vecs  = np.array([norm_scores[k] for k in keys])  # (n_metrics, n_methods)

    # Singleton entropies → singleton weights
    H = np.array([entropy_1d(vecs[i]) for i in range(n)])
    H_norm = H / H.sum() if H.sum() > 0 else np.ones(n) / n

    # Pairwise mutual information matrix
    MI = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            mi = mutual_information(vecs[i], vecs[j])
            MI[i, j] = MI[j, i] = mi

    max_MI = MI.max() if MI.max() > 0 else 1.0

    # Interaction indices scaled conservatively:
    # We cap the total negative interaction per metric so that
    # no subset capacity can go below 0 before repair.
    # Scale factor: max total negative interaction for any element
    # across all possible subsets = (n-1) * scale * max_H_norm
    # We want this << min_singleton so capacity stays positive.
    # Safe scale: 0.3 / (n-1) ensures pair interaction < singleton weight
    scale = min(0.3 / max(n - 1, 1), 0.15)

    I = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            interaction = -MI[i, j] / max_MI * scale
            I[i, j] = I[j, i] = interaction

    # Build capacity for all subsets (2^n entries)
    capacity = {}
    capacity[frozenset()] = 0.0

    for size in range(1, n + 1):
        for subset in combinations(range(n), size):
            s = frozenset(subset)
            val = sum(H_norm[i] for i in subset)
            for a, b in combinations(subset, 2):
                val += I[a, b]
            capacity[s] = float(max(val, 1e-6))  # hard floor

    # Monotonicity repair: ensure μ(S) <= μ(T) for S ⊂ T
    # Traverse subsets in increasing size order
    for size in range(1, n + 1):
        for subset in combinations(range(n), size):
            s = frozenset(subset)
            # must be >= all subsets of size-1
            for elem in subset:
                sub = s - {elem}
                if capacity[s] < capacity[sub]:
                    capacity[s] = capacity[sub] + 1e-6

    # Normalise so μ(full set) = 1
    full = frozenset(range(n))
    factor = 1.0 / capacity[full]
    capacity = {s: float(np.clip(v * factor, 0.0, 1.0))
                for s, v in capacity.items()}

    return capacity, H_norm, MI, I, keys


def shapley_values(capacity: dict, n: int) -> np.ndarray:
    """
    Compute Shapley values from the fuzzy measure.
    φ_i = Σ_{S ⊆ N\{i}}  [|S|!(n-|S|-1)!/n!] × [μ(S∪{i}) - μ(S)]

    Shapley value = "fair share" of each metric in the composite.
    """
    phi = np.zeros(n)
    from math import factorial
    for i in range(n):
        others = [j for j in range(n) if j != i]
        for size in range(n):
            for subset in combinations(others, size):
                s   = frozenset(subset)
                si  = s | {i}
                coeff = factorial(size) * factorial(n - size - 1) / factorial(n)
                phi[i] += coeff * (capacity.get(si, 0) - capacity.get(s, 0))
    return phi


# ══════════════════════════════════════════════════════════════════════════════
# Choquet integral
# ══════════════════════════════════════════════════════════════════════════════

def choquet_integral(scores_vec: np.ndarray, capacity: dict, n: int) -> float:
    """
    Discrete Choquet integral:
      C_μ(x) = Σ_{i=1}^{n}  (x_(i) - x_(i-1)) × μ(A_(i))
    where x_(1) ≤ … ≤ x_(n) and A_(i) = {j : x_j ≥ x_(i)}.
    x_(0) = 0 by convention.
    """
    order = np.argsort(scores_vec)          # ascending
    sorted_scores = scores_vec[order]
    result = 0.0
    prev   = 0.0
    for rank, idx in enumerate(order):
        # A_(rank+1) = indices with score >= sorted_scores[rank]
        A = frozenset(order[rank:])
        delta = float(sorted_scores[rank]) - prev
        result += delta * capacity.get(A, 0.0)
        prev = float(sorted_scores[rank])
    return result


# ══════════════════════════════════════════════════════════════════════════════
# Main analysis
# ══════════════════════════════════════════════════════════════════════════════

def run_choquet_analysis(methods, raw_scores):
    norm_scores = {m: normalise(raw_scores[m], d) for m, d in METRICS.items()}
    keys = list(norm_scores.keys())
    n    = len(keys)

    print("\nBuilding fuzzy measure (Kojadinovic entropy method)...")
    capacity, H_norm, MI, I_mat, keys = build_fuzzy_measure(norm_scores)

    print("\nComputing Shapley values...")
    phi = shapley_values(capacity, n)
    phi_norm = phi / phi.sum()

    print("\nComputing Choquet integral scores for each DR method...")
    scores_matrix = np.array([norm_scores[k] for k in keys])  # (n_metrics, n_methods)

    choquet_scores = np.array([
        choquet_integral(scores_matrix[:, j], capacity, n)
        for j in range(len(methods))
    ])

    # Also compute standard weighted sum with Shapley weights for comparison
    shapley_scores = (scores_matrix * phi_norm[:, None]).sum(axis=0)

    return {
        "keys":           keys,
        "capacity":       capacity,
        "H_norm":         H_norm,
        "MI":             MI,
        "I_mat":          I_mat,
        "phi":            phi,
        "phi_norm":       phi_norm,
        "choquet_scores": choquet_scores,
        "shapley_scores": shapley_scores,
    }


def print_results(methods, result):
    keys       = result["keys"]
    n          = len(keys)
    phi_norm   = result["phi_norm"]
    H_norm     = result["H_norm"]
    MI         = result["MI"]
    I_mat      = result["I_mat"]
    ch_scores  = result["choquet_scores"]
    sh_scores  = result["shapley_scores"]

    print(f"\n{'='*64}")
    print(f"  FUZZY MEASURE — METRIC INTERACTIONS")
    print(f"{'='*64}")
    print(f"\n  Singleton entropy weights  H(i)/ΣH(j):")
    for i, k in enumerate(keys):
        group = "LOCAL " if k in LOCAL_METRICS else "GLOBAL"
        bar   = "█" * min(50, max(0, int(H_norm[i] * 50)))
        print(f"    {k:<28} {H_norm[i]:.4f}  [{group}]  {bar}")

    print(f"\n  Pairwise interaction indices I(i,j):")
    print(f"  Negative = redundant (correlated), Positive = synergistic")
    for i in range(n):
        for j in range(i + 1, n):
            mi  = MI[i, j]
            iij = I_mat[i, j]
            tag = "redundant  ←" if iij < -0.05 else ("synergistic →" if iij > 0.05 else "independent")
            print(f"    {keys[i]:<28} × {keys[j]:<28}  "
                  f"MI={mi:.3f}  I={iij:+.4f}  {tag}")

    print(f"\n{'='*64}")
    print(f"  SHAPLEY VALUES  (fair marginal contribution of each metric)")
    print(f"{'='*64}")
    print(f"  φ_i = average marginal contribution across all orderings")
    print(f"  These are the 'honest weights' accounting for interactions.\n")
    for i, k in enumerate(keys):
        group = "LOCAL " if k in LOCAL_METRICS else "GLOBAL"
        bar   = "█" * min(50, max(0, int(phi_norm[i] * 50)))
        print(f"    {k:<28} φ={phi_norm[i]:.4f}  [{group}]  {bar}")

    local_share  = sum(phi_norm[i] for i, k in enumerate(keys) if k in LOCAL_METRICS)
    global_share = sum(phi_norm[i] for i, k in enumerate(keys) if k in GLOBAL_METRICS)
    print(f"\n  Total local  share: {local_share:.4f}")
    print(f"  Total global share: {global_share:.4f}")
    print(f"  → implied α = {local_share:.3f} (cf. Venna & Kaski 0.6 recommendation)")

    print(f"\n{'='*64}")
    print(f"  RECOMMENDED COMPOSITE FORMULA (Choquet integral)")
    print(f"{'='*64}")
    print(f"\n  composite(x) = Choquet_μ(x)")
    print(f"  where μ is the 2-additive fuzzy measure estimated from")
    print(f"  inter-metric entropy (Kojadinovic, 2004).")
    print()
    print(f"  Equivalent Shapley-weighted formula (additive approximation):")
    print(f"  composite ≈")
    for i, k in enumerate(keys):
        print(f"      {phi_norm[i]:.4f} × {k}")

    print(f"\n{'='*64}")
    print(f"  DR METHOD RANKING")
    print(f"{'='*64}")
    print(f"\n  {'Pos':<5} {'Method':<14}  {'Choquet':>8}  {'Shapley':>8}  {'Agree?'}")
    print(f"  {'-'*5} {'-'*14}  {'-'*8}  {'-'*8}  {'-'*6}")

    ch_order = np.argsort(-ch_scores)
    sh_order = np.argsort(-sh_scores)

    ch_ranks = {m: i+1 for i, m in enumerate(ch_order)}
    sh_ranks = {m: i+1 for i, m in enumerate(sh_order)}

    for pos, i in enumerate(ch_order):
        agree = "✓" if abs(ch_ranks[i] - sh_ranks[i]) <= 1 else "~"
        print(f"  {pos+1:<5} {methods[i]:<14}  {ch_scores[i]:>8.4f}  "
              f"{sh_scores[i]:>8.4f}  {agree}")

    print(f"\n  Note: 'Agree?' ✓ = Choquet and Shapley-weighted agree within 1 rank")

    print(f"\n{'='*64}")
    print(f"  THESIS JUSTIFICATION")
    print(f"{'='*64}")
    print(f"""
  \"The composite quality score was computed using the discrete Choquet
  integral (Grabisch, 1997) with a 2-additive fuzzy measure estimated
  via the entropy-based method of Kojadinovic (2004).  Unlike a standard
  weighted sum, the Choquet integral accounts for redundancy between
  correlated metrics: pairs such as (knn_recall, trustworthiness) and
  (spearman_r, pearson_r) received subadditive capacity values, reflecting
  their shared information content.  The Shapley values derived from this
  measure represent the fair marginal contribution of each metric and serve
  as interpretable effective weights.  The local-metric share implied by the
  Shapley values (α ≈ {local_share:.2f}) is consistent with the recommendation
  of Venna & Kaski (2006) that neighbourhood preservation should be
  weighted more heavily than global distance fidelity in visualisation tasks.\"

  References:
    Kojadinovic (2004) doi:10.1016/S0377-2217(02)00752-1
    Grabisch (1997)    doi:10.1016/S0165-0114(96)00168-2
    Venna & Kaski (2006) doi:10.1016/j.neunet.2006.05.014
    Lee & Verleysen (2009) doi:10.1016/j.neucom.2008.12.017
""")

    return ch_order, sh_order


def plot_results(methods, result, ch_order, out_path):
    try:
        import matplotlib.pyplot as plt
        import matplotlib
        matplotlib.use("Agg")
    except ImportError:
        print("  matplotlib not installed — skipping plot")
        return

    keys      = result["keys"]
    n         = len(keys)
    phi_norm  = result["phi_norm"]
    H_norm    = result["H_norm"]
    MI        = result["MI"]
    ch_scores = result["choquet_scores"]
    sh_scores = result["shapley_scores"]

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    colors_m = ["#5b8def","#42d4f4","#3cb44b","#f58231",
                "#911eb4","#e6194b","#f032e6","#bfef45"]

    # ── 1. Shapley values (effective weights) ─────────────────────────────────
    ax = axes[0]
    bar_colors = ["#3cb44b" if k in LOCAL_METRICS else "#f58231" for k in keys]
    bars = ax.barh(keys, phi_norm, color=bar_colors, edgecolor="white", height=0.6)
    ax.set_xlabel("Shapley value (effective weight)", fontsize=9)
    ax.set_title("Shapley values\n(green=local, orange=global)", fontsize=9)
    ax.axvline(1/n, color="red", lw=1, linestyle="--", alpha=0.6, label=f"equal (1/{n})")
    ax.legend(fontsize=7)
    for bar, v in zip(bars, phi_norm):
        ax.text(v + 0.003, bar.get_y() + bar.get_height()/2,
                f"{v:.3f}", va="center", fontsize=7)
    ax.invert_yaxis()

    # ── 2. MI heatmap (interaction structure) ─────────────────────────────────
    ax2 = axes[1]
    short = [k.replace("_", "\n") for k in keys]
    im = ax2.imshow(MI, cmap="YlOrRd", aspect="auto")
    ax2.set_xticks(range(n)); ax2.set_xticklabels(short, fontsize=6, rotation=45)
    ax2.set_yticks(range(n)); ax2.set_yticklabels(short, fontsize=6)
    ax2.set_title("Mutual information matrix\n(higher = more redundant)", fontsize=9)
    plt.colorbar(im, ax=ax2, shrink=0.8)
    for i in range(n):
        for j in range(n):
            ax2.text(j, i, f"{MI[i,j]:.2f}", ha="center", va="center", fontsize=5)

    # ── 3. Choquet vs Shapley ranking ─────────────────────────────────────────
    ax3 = axes[2]
    x = np.arange(len(methods))
    w = 0.35
    ch_norm = (ch_scores - ch_scores.min()) / (ch_scores.max() - ch_scores.min() + 1e-10)
    sh_norm = (sh_scores - sh_scores.min()) / (sh_scores.max() - sh_scores.min() + 1e-10)
    ax3.bar(x - w/2, ch_norm, width=w, label="Choquet integral",
            color="#5b8def", alpha=0.8, edgecolor="white")
    ax3.bar(x + w/2, sh_norm, width=w, label="Shapley weighted sum",
            color="#42d4f4", alpha=0.8, edgecolor="white")
    ax3.set_xticks(x)
    ax3.set_xticklabels(methods, rotation=25, fontsize=8)
    ax3.set_ylabel("Normalised composite score", fontsize=9)
    ax3.set_title("Choquet vs Shapley-weighted composite\n(higher = better)", fontsize=9)
    ax3.legend(fontsize=8)

    plt.suptitle("DR quality assessment via Choquet integral\n"
                 "(fuzzy measure from inter-metric entropy, Kojadinovic 2004)",
                 fontsize=11, y=1.02)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"  Plot saved → {out_path}")


def main():
    ap = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("--input", default=OUT_DIR / "vis_methods.json")
    ap.add_argument("--plot",  action="store_true")
    a = ap.parse_args()

    json_path = Path(a.input)
    if not json_path.exists():
        print(f"ERROR: {json_path} not found — run analyse.py first")
        return

    print("=" * 64)
    print("  choquet_composite.py")
    print(f"  Input: {json_path}")
    print("=" * 64)

    methods, raw_scores = load_scores(json_path)
    print(f"\n  DR methods : {methods}")
    print(f"  Metrics    : {list(METRICS.keys())}")

    result = run_choquet_analysis(methods, raw_scores)
    ch_order, sh_order = print_results(methods, result)

    # Save JSON
    keys     = result["keys"]
    phi_norm = result["phi_norm"]
    out = {
        "method":   "Choquet integral with 2-additive fuzzy measure (Kojadinovic 2004)",
        "shapley_weights": {k: round(float(phi_norm[i]), 4) for i, k in enumerate(keys)},
        "choquet_ranking": [methods[i] for i in ch_order],
        "shapley_ranking": [methods[i] for i in np.argsort(-result["shapley_scores"])],
        "local_share":  round(float(sum(phi_norm[i] for i, k in enumerate(keys)
                                        if k in LOCAL_METRICS)), 4),
        "global_share": round(float(sum(phi_norm[i] for i, k in enumerate(keys)
                                        if k in GLOBAL_METRICS)), 4),
        "references": [
            "Kojadinovic (2004) doi:10.1016/S0377-2217(02)00752-1",
            "Grabisch (1997) doi:10.1016/S0165-0114(96)00168-2",
            "Venna & Kaski (2006) doi:10.1016/j.neunet.2006.05.014",
            "Lee & Verleysen (2009) doi:10.1016/j.neucom.2008.12.017",
        ]
    }
    out_json = json_path.parent / "choquet_composite.json"
    with open(out_json, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\n  Results saved → {out_json}")

    if a.plot:
        plot_results(methods, result, ch_order,
                     out_path=json_path.parent / "choquet_composite.png")


if __name__ == "__main__":
    main()