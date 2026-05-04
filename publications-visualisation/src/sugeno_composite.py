"""
sugeno_composite.py
===================

Non-compensatory aggregation of DR quality metrics
using Sugeno integral and λ-fuzzy measure.

Key fix vs previous version
----------------------------
λ cannot be chosen freely when singletons sum to 1.0.
The condition g(Ω) = 1 requires solving:

    1 + λ = ∏ᵢ (1 + λ·wᵢ)

numerically for λ. Only λ=0 (additive measure) satisfies
this automatically when Σwᵢ = 1.

This script:
1. Loads metrics from analyse.py JSON
2. Normalizes metrics to [0,1]
3. For each target λ, rescales singletons so g(Ω) = 1
4. Validates g(Ω) = 1 for each λ
5. Computes Sugeno integral
6. Prints rankings + bottleneck diagnostics

Usage
-----
python sugeno_composite.py --input wyniki/methods.json [--show-metrics]
"""

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.optimize import brentq


# ──────────────────────────────────────────────────────────────────────────────
# NORMALIZATION
# ──────────────────────────────────────────────────────────────────────────────

def normalize_metrics(cm):
    """Normalize all metrics to [0,1], higher = better."""
    return {
        "knn":      cm["knn_recall"],
        "trust":    cm["trustworthiness"],
        "nh":       cm["neighborhood_hit"],
        "spearman": (cm["spearman_r"] + 1) / 2,
        "pearson":  (cm["pearson_r"] + 1) / 2,
        "spde":     cm["distance_preservation"],
        "stress":   1 / (1 + cm["kruskal_stress"]),
    }


# ──────────────────────────────────────────────────────────────────────────────
# SINGLETON WEIGHTS
# ──────────────────────────────────────────────────────────────────────────────

SINGLETONS = {
    "knn":      0.17,
    "trust":    0.17,
    "nh":       0.16,
    "spearman": 0.15,
    "pearson":  0.15,
    "spde":     0.10,
    "stress":   0.10,
}


# ──────────────────────────────────────────────────────────────────────────────
# λ-FUZZY MEASURE
# ──────────────────────────────────────────────────────────────────────────────

def fuzzy_measure(subset, weights, lambd):
    """
    Iterative λ-fuzzy measure (Sugeno 1974).
    g(A ∪ {x}) = g(A) + g({x}) + λ · g(A) · g({x})
    Commutative — order of iteration does not affect result.
    """
    subset = list(subset)
    if len(subset) == 0:
        return 0.0
    if len(subset) == 1:
        return weights[subset[0]]
    g = weights[subset[0]]
    for item in subset[1:]:
        w = weights[item]
        g = g + w + lambd * g * w
    return g


def validate_measure(weights, lambd, tol=1e-6):
    """Assert g(Ω) = 1.0. Raises ValueError if violated."""
    all_keys = list(weights.keys())
    g_omega = fuzzy_measure(all_keys, weights, lambd)
    if abs(g_omega - 1.0) > tol:
        raise ValueError(
            f"g(Ω) = {g_omega:.8f} ≠ 1.0 for λ = {lambd:.6f}"
        )
    return g_omega


def scale_weights_for_lambda(weights, lambd, tol=1e-10):
    """
    Scale singleton weights by scalar α so that g(Ω; α·wᵢ, λ) = 1.
    Preserves relative importance of metrics.
    """
    if abs(lambd) < 1e-12:
        total = sum(weights.values())
        return {k: v / total for k, v in weights.items()}

    def equation(alpha):
        scaled = {k: alpha * v for k, v in weights.items()}
        return fuzzy_measure(list(scaled.keys()), scaled, lambd) - 1.0

    try:
        alpha = brentq(equation, 1e-9, 100.0, xtol=tol)
    except ValueError:
        return weights

    return {k: alpha * v for k, v in weights.items()}


# ──────────────────────────────────────────────────────────────────────────────
# SUGENO INTEGRAL
# ──────────────────────────────────────────────────────────────────────────────

def sugeno_integral(metrics, weights, lambd):
    """
    Sugeno integral: S_g(x) = max_i min( x_(i), g(A_i) )
    x_(i) : metric values sorted ascending
    A_i   : tail set { x_(i), ..., x_(n) }
    """
    items  = sorted(metrics.items(), key=lambda x: x[1])
    values = [v for _, v in items]
    names  = [k for k, _ in items]

    sugeno_vals = []
    for i in range(len(values)):
        subset = names[i:]
        g = fuzzy_measure(subset, weights, lambd)
        sugeno_vals.append(min(values[i], g))

    return max(sugeno_vals)


# ──────────────────────────────────────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description="Sugeno composite score for DR methods")
    ap.add_argument("--input", required=True, help="JSON output from analyse.py")
    ap.add_argument("--show-metrics", action="store_true",
                    help="Print normalized metric values for each method")
    args = ap.parse_args()

    with open(Path(args.input), encoding="utf-8") as f:
        data = json.load(f)
    methods = data["methods"]

    w_sum = sum(SINGLETONS.values())
    print(f"Singleton weights sum: {w_sum:.6f}")
    print("Note: when Σwᵢ=1, only λ=0 gives g(Ω)=1 automatically.")
    print("      For λ≠0, singletons are rescaled (α·wᵢ) preserving relative importance.")

    # λ<0 = redundancy (substitutive metrics), λ>0 = synergy (complementary)
    target_lambdas = [0.0, -0.3, -0.6, -0.9, 0.3, 0.6]

    for lambd in target_lambdas:
        print("\n" + "=" * 90)
        print(f"SUGENO COMPOSITE — λ = {lambd}")
        print("=" * 90)

        scaled_weights = scale_weights_for_lambda(SINGLETONS, lambd)

        try:
            g_omega = validate_measure(scaled_weights, lambd)
            print(f"g(Ω) = {g_omega:.8f}  ✓")
            print(f"Scaled weights: { {k: round(v,4) for k,v in scaled_weights.items()} }")
        except ValueError as e:
            print(f"  ✗ VALIDATION FAILED: {e}")
            continue

        results = []

        if args.show_metrics:
            print(f"\n  {'Method':<14}", end="")
            for k in SINGLETONS:
                print(f"  {k:>8}", end="")
            print()

        for key, method_data in methods.items():
            metrics = normalize_metrics(method_data["custom_metric"])

            if args.show_metrics:
                print(f"  {method_data['short']:<14}", end="")
                for k in SINGLETONS:
                    print(f"  {metrics[k]:>8.3f}", end="")
                print()

            score = sugeno_integral(metrics, scaled_weights, lambd)
            results.append({
                "method":  method_data["short"],
                "score":   score,
                "metrics": metrics,
            })

        ranked = sorted(results, key=lambda x: x["score"], reverse=True)

        print(f"\n{'Rank':<6} {'Method':<14} {'Sugeno':>10}  {'bottleneck_val':>14}  {'bottleneck_metric'}")
        print(f"{'-'*6} {'-'*14} {'-'*10}  {'-'*14}  {'-'*20}")

        for rank, r in enumerate(ranked, 1):
            bn_key = min(r["metrics"], key=r["metrics"].get)
            bn_val = r["metrics"][bn_key]
            print(
                f"{rank:<6} {r['method']:<14} {r['score']:>10.4f}  "
                f"{bn_val:>14.4f}  {bn_key}"
            )

    print("\n" + "=" * 90)
    print("DIAGNOSTIC NOTE")
    print("  Sugeno integral is a bottleneck aggregator: score ≤ min(metric values).")
    print("  Methods with similar worst-case metrics → similar/identical scores.")
    print("  If discrimination remains poor across all λ, use Choquet integral instead.")
    print("=" * 90)


if __name__ == "__main__":
    main()