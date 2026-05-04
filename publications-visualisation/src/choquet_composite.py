"""
choquet_composite.py
====================

Continuous non-additive aggregation of DR quality metrics
using a validated 2-additive Choquet integral.

Key properties
---------------
✓ normalized metrics in [0,1]
✓ continuous aggregation (better discrimination than Sugeno)
✓ pairwise redundancy/synergy modeling
✓ validated fuzzy capacity:
    - g(∅)=0
    - g(Ω)=1
    - monotonicity
✓ interpretable Shapley values

Usage
-----
python choquet_composite.py --input ../output/vis_methods.json
"""

import argparse
import itertools
import json
from pathlib import Path


# ──────────────────────────────────────────────────────────────────────────────
# NORMALIZATION
# ──────────────────────────────────────────────────────────────────────────────

def normalize_metrics(cm):
    """
    Normalize all metrics to [0,1]
    and ensure higher = better.
    """

    return {

        # local structure
        "knn": cm["knn_recall"],

        "trust": cm["trustworthiness"],

        "nh": cm["neighborhood_hit"],

        # global structure
        "spearman": (cm["spearman_r"] + 1) / 2,

        "pearson": (cm["pearson_r"] + 1) / 2,

        "spde": cm["distance_preservation"],

        # lower stress = better
        "stress": 1 / (1 + cm["kruskal_stress"]),
    }


# ──────────────────────────────────────────────────────────────────────────────
# SINGLETON IMPORTANCE
# must sum to 1.0
# ──────────────────────────────────────────────────────────────────────────────

SINGLETONS = {

    # local structure
    "knn": 0.17,
    "trust": 0.17,
    "nh": 0.16,

    # global structure
    "spearman": 0.15,
    "pearson": 0.15,
    "spde": 0.10,
    "stress": 0.10,
}


# ──────────────────────────────────────────────────────────────────────────────
# PAIRWISE INTERACTIONS
#
# positive  -> synergy
# negative  -> redundancy
#
# IMPORTANT:
# Σ interactions must equal 0
# so that g(Ω)=1 remains satisfied.
# ──────────────────────────────────────────────────────────────────────────────

INTERACTIONS = {

    # local metrics partially redundant
    frozenset(["knn", "trust"]): -0.04,

    # rank correlations partially redundant
    frozenset(["spearman", "pearson"]): -0.03,

    # local neighborhood synergy
    frozenset(["knn", "nh"]): 0.03,

    # trust + global geometry synergy
    frozenset(["trust", "spde"]): 0.04,
}


# ──────────────────────────────────────────────────────────────────────────────
# 2-ADDITIVE FUZZY MEASURE
# ──────────────────────────────────────────────────────────────────────────────

def fuzzy_measure(subset):
    """
    2-additive fuzzy capacity.

    g(A)
    =
    Σ singleton weights
    +
    Σ pairwise interactions

    Only interactions fully contained in A contribute.
    """

    subset = set(subset)

    # explicit empty-set handling
    if len(subset) == 0:
        return 0.0

    val = 0.0

    # singleton contributions
    for k in subset:
        val += SINGLETONS[k]

    # pairwise interactions
    for pair, interaction in INTERACTIONS.items():

        if pair.issubset(subset):
            val += interaction

    return val


# ──────────────────────────────────────────────────────────────────────────────
# VALIDATION
# ──────────────────────────────────────────────────────────────────────────────

def validate_capacity():
    """
    Validate fuzzy measure axioms:

    1. g(∅)=0
    2. g(Ω)=1
    3. monotonicity
    """

    keys = list(SINGLETONS.keys())

    g_empty = fuzzy_measure([])
    g_full  = fuzzy_measure(keys)

    print("\nCapacity validation")
    print("-" * 40)

    print(f"g(∅) = {g_empty:.6f}")
    print(f"g(Ω) = {g_full:.6f}")

    if abs(g_empty) > 1e-9:
        raise ValueError(
            "Invalid fuzzy measure: g(∅) != 0"
        )

    if abs(g_full - 1.0) > 1e-6:
        raise ValueError(
            f"Invalid fuzzy measure: g(Ω)={g_full:.6f} != 1"
        )

    # exhaustive monotonicity check
    # O(2^n) but feasible for n=7
    for r1 in range(len(keys)):
        for r2 in range(r1 + 1, len(keys) + 1):

            for A in itertools.combinations(keys, r1):
                for B in itertools.combinations(keys, r2):

                    A = set(A)
                    B = set(B)

                    if A.issubset(B):

                        if fuzzy_measure(A) > fuzzy_measure(B) + 1e-9:

                            raise ValueError(
                                f"Monotonicity violated:\n"
                                f"{A} ⊄ {B}\n"
                                f"g(A)={fuzzy_measure(A)} > g(B)={fuzzy_measure(B)}"
                            )

    print("Monotonicity: OK")


# ──────────────────────────────────────────────────────────────────────────────
# CHOQUET INTEGRAL
# ──────────────────────────────────────────────────────────────────────────────

def choquet_integral(metrics):
    """
    Discrete Choquet integral.

    C_g(x)
    =
    Σ (x_i - x_{i-1}) g(A_i)

    where:
        x_i sorted ascending
        A_i = tail subsets
    """

    items = sorted(metrics.items(), key=lambda x: x[1])

    values = [v for _, v in items]
    names  = [k for k, _ in items]

    total = 0.0
    prev  = 0.0

    for i in range(len(values)):

        delta = values[i] - prev

        subset = names[i:]

        total += delta * fuzzy_measure(subset)

        prev = values[i]

    return total


# ──────────────────────────────────────────────────────────────────────────────
# SHAPLEY VALUES
# ──────────────────────────────────────────────────────────────────────────────

def shapley_values():
    """
    Shapley importance values
    for 2-additive capacity.

    φ_i
    =
    w_i
    +
    1/2 Σ I(i,j)
    """

    vals = {}

    for i in SINGLETONS:

        phi = SINGLETONS[i]

        for pair, interaction in INTERACTIONS.items():

            if i in pair:
                phi += 0.5 * interaction

        vals[i] = phi

    return vals


# ──────────────────────────────────────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────────────────────────────────────

def main():

    ap = argparse.ArgumentParser(
        description="Choquet composite score for DR methods"
    )

    ap.add_argument(
        "--input",
        required=True,
        help="JSON output from analyse.py"
    )

    args = ap.parse_args()

    with open(Path(args.input), encoding="utf-8") as f:
        data = json.load(f)

    methods = data["methods"]

    print("\n" + "=" * 90)
    print("CHOQUET COMPOSITE SCORE")
    print("=" * 90)

    # validate capacity
    validate_capacity()

    print("\nSingleton weights")
    print("-" * 40)

    for k, v in SINGLETONS.items():
        print(f"{k:<12} {v:.4f}")

    print("\nPairwise interactions")
    print("-" * 40)

    for pair, val in INTERACTIONS.items():

        p = list(pair)

        relation = (
            "synergy"
            if val > 0
            else "redundancy"
        )

        print(
            f"({p[0]}, {p[1]}) "
            f"{val:+.4f} "
            f"[{relation}]"
        )

    # compute scores
    results = []

    print("\n" + "=" * 90)
    print(f"{'Method':<14} {'Choquet':>10}")
    print("=" * 90)

    for key, method_data in methods.items():

        metrics = normalize_metrics(
            method_data["custom_metric"]
        )

        score = choquet_integral(metrics)

        results.append({
            "method": method_data["short"],
            "score": score,
            "metrics": metrics,
        })

        print(
            f"{method_data['short']:<14} "
            f"{score:>10.4f}"
        )

    # ranking
    ranked = sorted(
        results,
        key=lambda x: x["score"],
        reverse=True
    )

    print("\n-- Ranking by Choquet Integral --")

    for rank, r in enumerate(ranked, 1):

        print(
            f"{rank}. "
            f"{r['method']:<14} "
            f"Choquet={r['score']:.4f}"
        )

    # shapley values
    print("\n" + "=" * 90)
    print("SHAPLEY VALUES")
    print("=" * 90)

    shap = shapley_values()

    for k, v in sorted(
        shap.items(),
        key=lambda x: x[1],
        reverse=True
    ):

        print(
            f"{k:<12} "
            f"{v:.4f}"
        )

    print("\n" + "=" * 90)
    print("INTERPRETATION")
    print("- Choquet preserves continuous metric differences.")
    print("- Negative interactions reduce redundancy.")
    print("- Positive interactions reward complementary behavior.")
    print("- Shapley values quantify effective metric importance.")
    print("=" * 90)


if __name__ == "__main__":
    main()