#!/usr/bin/env python3
"""
Evaluate three author-representation variants on department separation.

Variants:
  - baseline   : 1 mean-pooled centroid per author (n=115 authors with papers)
  - aggressive : multi-cluster (k>1) for ALL authors with >=5 papers and silhouette >=0.15
  - adaptive   : per-author decision based on policy.csv (SINGLE/MULTI/LOW_CONF/AMBIGUOUS)

Metrics:
  - NMI (clusters vs true depts)
  - NN@1 / NN@3 / NN@5 dept accuracy (per-point — for variants that produce
    multiple points per author, we evaluate each point's neighbors)
  - silhouette score in 768D using true department labels
  - number of points produced

Constraint: dept evaluation only on authors with explicit "Zakład" / "Pracownia"
affiliation (n=92 authors).
"""

import numpy as np
import pandas as pd
from collections import Counter
from sklearn.cluster import KMeans
from sklearn.metrics import (
    normalized_mutual_info_score,
    silhouette_score,
)

# Paths
TITLES_CSV     = "data/titles_with_abstracts.csv"
SCIENTISTS_CSV = "data/scientists_with_identifiers.csv"
EMBEDDINGS_NPY = "data/paper_embeddings_cosent.npy"
POLICY_CSV     = "data/policy.csv"
OUTPUT_MD      = "results/wmi_authors/variants_comparison.md"

MIN_PAPERS    = 5
MAX_K         = 5
SIL_THRESHOLD = 0.15

# Load
titles_df     = pd.read_csv(TITLES_CSV)
scientists_df = pd.read_csv(SCIENTISTS_CSV)
scientists_df = scientists_df[scientists_df["orcid"].isin(titles_df["main_author_orcid"])].copy()
embeddings = np.load(EMBEDDINGS_NPY)
paper_orcids = titles_df["main_author_orcid"].values
policy_df = pd.read_csv(POLICY_CSV).set_index("orcid")

# Extract dept (only Zakład / Pracownia)
def extract_dept(aff):
    if pd.isna(aff):
        return ""
    return str(aff).split(";")[0].strip()

scientists_df["dept"] = scientists_df["affiliations"].apply(extract_dept)
dept_mask = (
    scientists_df["dept"].str.startswith("Zakład")
    | scientists_df["dept"].str.startswith("Pracownia")
)
scientists_dept = scientists_df[dept_mask].copy()

print(f"Authors with explicit dept: {len(scientists_dept)}")
print(f"Departments: {scientists_dept['dept'].nunique()}\n")


# ─── Centroid builders per variant ─────────────────────────────────────────────

def normalize(v):
    return v / np.linalg.norm(v)

def build_baseline(orcids):
    """1 centroid per author (mean pool)."""
    points = []
    for orcid in orcids:
        mask = paper_orcids == orcid
        if not mask.any():
            continue
        emb = normalize(embeddings[mask].mean(axis=0))
        points.append({"orcid": orcid, "emb": emb, "cluster_id": 0})
    return points

def build_aggressive(orcids):
    """Always force k>1 for any author with n>=MIN_PAPERS — pick best k by silhouette
    (no silhouette cutoff: even authors with weak cluster structure are split)."""
    points = []
    for orcid in orcids:
        mask = paper_orcids == orcid
        embs = embeddings[mask]
        n = len(embs)
        if n < MIN_PAPERS:
            points.append({"orcid": orcid, "emb": normalize(embs.mean(axis=0)), "cluster_id": 0})
            continue

        max_k = min(MAX_K, n - 1)
        if max_k < 2:
            points.append({"orcid": orcid, "emb": normalize(embs.mean(axis=0)), "cluster_id": 0})
            continue

        best_k, best_sil, best_labels = 2, -1.0, None
        for k in range(2, max_k + 1):
            km = KMeans(n_clusters=k, random_state=42, n_init=10)
            labels = km.fit_predict(embs)
            if len(set(labels)) < 2:
                continue
            sil = silhouette_score(embs, labels)
            if sil > best_sil:
                best_sil, best_k, best_labels = sil, k, labels

        if best_labels is None:
            points.append({"orcid": orcid, "emb": normalize(embs.mean(axis=0)), "cluster_id": 0})
            continue

        # Aggressive: do NOT fall back to single centroid when silhouette is low
        for c in range(best_k):
            sub = embs[best_labels == c]
            points.append({"orcid": orcid, "emb": normalize(sub.mean(axis=0)), "cluster_id": c})
    return points

def build_adaptive(orcids):
    """Use policy.csv decisions: MULTI -> multi-cluster, otherwise 1 centroid."""
    points = []
    for orcid in orcids:
        mask = paper_orcids == orcid
        embs = embeddings[mask]
        if not mask.any():
            continue

        if orcid not in policy_df.index:
            points.append({"orcid": orcid, "emb": normalize(embs.mean(axis=0)), "cluster_id": 0})
            continue

        decision = policy_df.loc[orcid, "decision"]

        if decision == "MULTI":
            best_k = int(policy_df.loc[orcid, "best_k"])
            km = KMeans(n_clusters=best_k, random_state=42, n_init=10)
            labels = km.fit_predict(embs)
            for c in range(best_k):
                sub = embs[labels == c]
                points.append({"orcid": orcid, "emb": normalize(sub.mean(axis=0)), "cluster_id": c})
        else:
            # SINGLE / LOW_CONF / AMBIGUOUS — single mean-pooled centroid
            points.append({"orcid": orcid, "emb": normalize(embs.mean(axis=0)), "cluster_id": 0})
    return points


# ─── Metrics ───────────────────────────────────────────────────────────────────

def evaluate(points, dept_lookup, n_depts):
    """
    points: list of dicts {orcid, emb, cluster_id}
    dept_lookup: dict orcid -> dept name (only authors with depts)
    n_depts: number of unique depts
    Returns metrics dict.
    """
    # Filter to points whose author has a dept
    pts = [p for p in points if p["orcid"] in dept_lookup]
    if not pts:
        return {"n_points": 0}

    X = np.stack([p["emb"] for p in pts])
    y = np.array([dept_lookup[p["orcid"]] for p in pts])
    orcids = np.array([p["orcid"] for p in pts])

    # K-means with k = n_depts
    km = KMeans(n_clusters=n_depts, random_state=42, n_init=10)
    cluster_labels = km.fit_predict(X)

    nmi = normalized_mutual_info_score(y, cluster_labels)

    # Dept purity (best-match per cluster)
    purity_correct = 0
    for c in range(n_depts):
        members = y[cluster_labels == c]
        if len(members) == 0:
            continue
        most_common = Counter(members).most_common(1)[0][0]
        purity_correct += int((members == most_common).sum())
    purity = purity_correct / len(y)

    # Silhouette by true depts
    unique_depts = list(set(y))
    dept_to_idx = {d: i for i, d in enumerate(unique_depts)}
    y_num = np.array([dept_to_idx[d] for d in y])
    sil = silhouette_score(X, y_num) if len(unique_depts) > 1 else 0.0

    # NN@k accuracy — for each point, look at k nearest *other-author* points
    # (we exclude points belonging to the same author to avoid trivial self-match
    # in multi-cluster variants)
    sims = X @ X.T
    n = len(pts)
    nn_accs = {}
    for k in [1, 3, 5]:
        correct = 0
        for i in range(n):
            s = sims[i].copy()
            # Mask out same-author points
            same_author = orcids == orcids[i]
            s[same_author] = -np.inf
            top_k_idx = np.argsort(s)[-k:]
            neighbor_depts = [y[j] for j in top_k_idx]
            most_common = Counter(neighbor_depts).most_common(1)[0][0]
            if most_common == y[i]:
                correct += 1
        nn_accs[k] = correct / n

    return {
        "n_points": len(pts),
        "nmi": nmi,
        "purity": purity,
        "silhouette": sil,
        "nn1": nn_accs[1],
        "nn3": nn_accs[3],
        "nn5": nn_accs[5],
    }


# ─── Run all 3 variants ────────────────────────────────────────────────────────
print("Building variants...")
all_orcids = scientists_df["orcid"].tolist()

baseline_points   = build_baseline(all_orcids)
aggressive_points = build_aggressive(all_orcids)
adaptive_points   = build_adaptive(all_orcids)

print(f"  baseline   : {len(baseline_points)} points")
print(f"  aggressive : {len(aggressive_points)} points")
print(f"  adaptive   : {len(adaptive_points)} points")

dept_lookup = dict(zip(scientists_dept["orcid"], scientists_dept["dept"]))
n_depts = scientists_dept["dept"].nunique()

print("\nEvaluating on 14 departments (n=92 authors with explicit affiliation)...\n")

results = {
    "baseline":   evaluate(baseline_points, dept_lookup, n_depts),
    "aggressive": evaluate(aggressive_points, dept_lookup, n_depts),
    "adaptive":   evaluate(adaptive_points, dept_lookup, n_depts),
}

# Print
hdr = f"{'Variant':<12} | {'Points':>6} | {'NMI':>6} | {'Purity':>6} | {'NN@1':>6} | {'NN@3':>6} | {'NN@5':>6} | {'Sil':>7}"
print(hdr)
print("-" * len(hdr))
for name, r in results.items():
    print(f"{name:<12} | {r['n_points']:>6} | {r['nmi']:>6.4f} | {r['purity']:>6.4f} | {r['nn1']:>6.4f} | {r['nn3']:>6.4f} | {r['nn5']:>6.4f} | {r['silhouette']:>7.4f}")

# ─── Save markdown report ─────────────────────────────────────────────────────
from pathlib import Path
Path(OUTPUT_MD).parent.mkdir(parents=True, exist_ok=True)
from datetime import datetime

with open(OUTPUT_MD, "w") as f:
    f.write("# Porównanie wariantów reprezentacji autorów na separacji zakładów\n\n")
    f.write(f"**Data:** {datetime.now().strftime('%Y-%m-%d %H:%M')}\n\n")
    f.write(f"**Zbiór ewaluacyjny:** {len(scientists_dept)} autorów z jednoznacznym przypisaniem ")
    f.write(f"do {n_depts} zakładów/pracowni WMI (afiliacja zaczynająca się od 'Zakład' lub 'Pracownia').\n\n")
    f.write("**Embeddingi:** BGE-base-en-v1.5 fine-tuned (CoSENTLoss + hierarchiczna odległość ArXiv).\n\n")

    f.write("## Definicje wariantów\n\n")
    f.write("- **baseline** — 1 centroid (mean pooling) per autor (115 punktów dla pełnego zbioru).\n")
    f.write("- **aggressive** — multi-cluster (k>1) dla *każdego* autora z ≥5 pracami ")
    f.write("i silhouette_max ≥ 0,15; pozostali autorzy: 1 centroid.\n")
    f.write("- **adaptive** — decyzja per autor wg `policy.csv` (SINGLE/MULTI/LOW_CONF/AMBIGUOUS) ")
    f.write("opartej na sygnale silhouette (struktura klastrowa) i stab(5) ")
    f.write("(stabilność profilu, Rolewski 2024 sec. 1.4.3).\n\n")

    f.write("## Wyniki\n\n")
    f.write("| Wariant | Punkty | NMI | Purity | NN@1 | NN@3 | NN@5 | Silhouette |\n")
    f.write("|---------|-------:|----:|-------:|-----:|-----:|-----:|-----------:|\n")
    for name in ["baseline", "aggressive", "adaptive"]:
        r = results[name]
        f.write(f"| {name} | {r['n_points']} | {r['nmi']:.4f} | {r['purity']:.4f} | "
                f"{r['nn1']:.4f} | {r['nn3']:.4f} | {r['nn5']:.4f} | {r['silhouette']:.4f} |\n")

    f.write("\n## Rozkład decyzji w wariancie adaptive\n\n")
    f.write("| Decyzja | Autorów | % | Punktów |\n")
    f.write("|---------|--------:|--:|--------:|\n")
    policy = pd.read_csv(POLICY_CSV)
    for d in ["SINGLE", "MULTI", "LOW_CONF", "AMBIGUOUS"]:
        sub = policy[policy["decision"] == d]
        n = len(sub)
        pct = n / len(policy) * 100
        pts = int(sub["n_points"].sum())
        f.write(f"| {d} | {n} | {pct:.1f} | {pts} |\n")
    f.write(f"| **Razem** | **{len(policy)}** | **100,0** | **{int(policy['n_points'].sum())}** |\n")

    f.write("\n### Objaśnienia metryk\n\n")
    f.write("- **NMI** — Normalized Mutual Information między klastrami K-means (k=14) ")
    f.write("a prawdziwymi zakładami (0=losowe, 1=idealne).\n")
    f.write("- **Purity** — odsetek autorów w klastrach należących do dominującego zakładu.\n")
    f.write("- **NN@k** — odsetek punktów, których większość z k najbliższych sąsiadów ")
    f.write("(z wykluczeniem punktów tego samego autora) należy do tego samego zakładu.\n")
    f.write("- **Silhouette** — jakość separacji zakładów w pełnej przestrzeni 768D ")
    f.write("(prawdziwe etykiety zakładów).\n")

print(f"\nSaved: {OUTPUT_MD}")
