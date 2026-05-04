"""
trimmed_composite.py
====================

Computes a robust Composite Score using a trimmed mean aggregation.

Pipeline:
1. Load JSON produced by analyse.py
2. Extract projection quality metrics
3. Normalize all metrics to [0,1]
4. Remove best and worst metric for each method
5. Compute trimmed mean from remaining metrics
6. Print ranking

Purpose
-------
This step reduces the influence of extreme metric values while still
remaining an averaging-based aggregation method.

It serves as an intermediate step between:
- arithmetic mean (fully compensatory)
- Sugeno integral (non-compensatory)

Usage
-----
python trimmed_composite.py --input wyniki/methods.json
"""

import argparse
import json
from pathlib import Path

import numpy as np


# ──────────────────────────────────────────────────────────────────────────────
# Normalization
# ──────────────────────────────────────────────────────────────────────────────

def normalize_metrics(cm):

    return {

        "knn": cm["knn_recall"],

        "trust": cm["trustworthiness"],

        "nh": cm["neighborhood_hit"],

        # [-1,1] -> [0,1]
        "spearman": (cm["spearman_r"] + 1) / 2,

        # [-1,1] -> [0,1]
        "pearson": (cm["pearson_r"] + 1) / 2,

        "spde": cm["distance_preservation"],

        # smooth inverse stress normalization
        "stress": 1 / (1 + cm["kruskal_stress"]),
    }


# ──────────────────────────────────────────────────────────────────────────────
# Trimmed mean
# ──────────────────────────────────────────────────────────────────────────────

def trimmed_mean(values):
    """
    Remove min and max value,
    compute mean from remaining values.
    """

    vals = sorted(values)

    # remove smallest and largest
    trimmed = vals[1:-1]

    return float(np.mean(trimmed))


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main():

    ap = argparse.ArgumentParser(
        description="Compute trimmed-mean composite score"
    )

    ap.add_argument(
        "--input",
        required=True,
        help="JSON output from analyse.py"
    )

    args = ap.parse_args()

    input_path = Path(args.input)

    with open(input_path, encoding="utf-8") as f:
        data = json.load(f)

    methods = data["methods"]

    results = []

    print("\n" + "=" * 90)
    print("TRIMMED MEAN COMPOSITE SCORE")
    print("=" * 90)

    print(
        f"{'Method':<14} "
        f"{'Trimmed':>10} "
        f"{'Removed min/max':>25}"
    )

    print(
        f"{'-'*14} "
        f"{'-'*10} "
        f"{'-'*25}"
    )

    for key, method_data in methods.items():

        cm = method_data["custom_metric"]

        metrics = normalize_metrics(cm)

        vals = list(metrics.values())

        sorted_vals = sorted(vals)

        removed = (
            round(sorted_vals[0], 4),
            round(sorted_vals[-1], 4)
        )

        score = trimmed_mean(vals)

        results.append({
            "method": method_data["short"],
            "score": score,
            "removed": removed,
        })

        print(
            f"{method_data['short']:<14} "
            f"{score:>10.4f} "
            f"{str(removed):>25}"
        )

    # ranking
    ranked = sorted(results, key=lambda x: x["score"], reverse=True)

    print("\n-- Ranking by Trimmed Mean Composite --")

    for rank, r in enumerate(ranked, 1):

        print(
            f"{rank}. "
            f"{r['method']:<14} "
            f"Trimmed={r['score']:.4f}"
        )


if __name__ == "__main__":
    main()