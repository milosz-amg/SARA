"""
metric_correlations.py
======================

Oblicza korelacje Pearsona między znormalizowanymi metrykami jakości
projekcji dla wszystkich metod DR.

Uzasadnia (lub podważa) założone interakcje w 2-additive fuzzy measure.

Usage
-----
python metric_correlations.py --input ../output/vis_methods.json
"""

import argparse
import json
from pathlib import Path

import numpy as np


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


INTERACTIONS = {
    ("knn", "trust"):        -0.04,
    ("spearman", "pearson"): -0.03,
    ("knn", "nh"):           +0.03,
    ("trust", "spde"):       +0.04,
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    args = ap.parse_args()

    with open(args.input, encoding="utf-8") as f:
        data = json.load(f)

    methods = data["methods"]
    keys    = ["knn", "trust", "nh", "spearman", "pearson", "spde", "stress"]

    # zbierz znormalizowane metryki dla wszystkich metod
    rows = []
    names = []
    for k, md in methods.items():
        nm = normalize_metrics(md["custom_metric"])
        rows.append([nm[key] for key in keys])
        names.append(md["short"])

    X = np.array(rows)  # shape (n_methods, n_metrics)

    print("Normalized metric values per method:")
    print(f"  {'Method':<14}", end="")
    for k in keys:
        print(f"  {k:>8}", end="")
    print()
    print("  " + "-" * (14 + 10 * len(keys)))
    for i, name in enumerate(names):
        print(f"  {name:<14}", end="")
        for v in X[i]:
            print(f"  {v:>8.4f}", end="")
        print()

    # macierz korelacji Pearsona między metrykami
    # każda kolumna = wektor wartości danej metryki po metodach
    corr = np.corrcoef(X.T)  # shape (n_metrics, n_metrics)

    print("\n" + "=" * 70)
    print("PEARSON CORRELATION MATRIX (between metrics, across DR methods)")
    print("=" * 70)
    print(f"  {'':>10}", end="")
    for k in keys:
        print(f"  {k:>8}", end="")
    print()
    print("  " + "-" * (10 + 10 * len(keys)))
    for i, ki in enumerate(keys):
        print(f"  {ki:>10}", end="")
        for j in range(len(keys)):
            v = corr[i, j]
            marker = " *" if abs(v) > 0.7 and i != j else "  "
            print(f"  {v:>7.3f}{marker[0]}", end="")
        print()

    print("\n  (* = |r| > 0.7 — wysoka korelacja)")

    # sprawdź założone interakcje
    print("\n" + "=" * 70)
    print("VALIDATION OF ASSUMED INTERACTIONS")
    print("=" * 70)
    print(f"  {'Para metryk':<25} {'Korelacja':>10}  {'Założona interakcja':>20}  Ocena")
    print("  " + "-" * 75)

    for (a, b), interaction in INTERACTIONS.items():
        i, j   = keys.index(a), keys.index(b)
        r      = corr[i, j]
        kind   = "redundancja" if interaction < 0 else "synergia"
        # redundancja uzasadniona gdy wysoka dodatnia korelacja
        # synergia uzasadniona gdy niska korelacja (metryki mierzą różne rzeczy)
        if interaction < 0:
            ok = "✓ uzasadniona" if r > 0.5 else ("△ słabe podstawy" if r > 0 else "✗ nieuzasadniona")
        else:
            ok = "✓ uzasadniona" if r < 0.5 else ("△ słabe podstawy" if r < 0.7 else "✗ nieuzasadniona")

        print(f"  ({a}, {b}){'':<{20-len(a)-len(b)}} {r:>10.3f}  "
              f"{interaction:>+6.2f} [{kind}]{'':<10}  {ok}")

    print()
    print("Interpretacja:")
    print("  Redundancja (I<0) uzasadniona gdy metryki silnie skorelowane (r>0.5)")
    print("  Synergia    (I>0) uzasadniona gdy metryki słabo skorelowane  (r<0.5)")


if __name__ == "__main__":
    main()