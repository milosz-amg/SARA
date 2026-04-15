#!/usr/bin/env python3
"""
Evaluate how well embeddings separate WMI departments.

Compares original BGE-base vs CoSENT fine-tuned model on:
- Global metrics: intra/inter-dept cosine similarity, NN accuracy, clustering
- Per-department breakdown: cohesion, nearest other dept, separation score
"""

import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
from sklearn.metrics import (
    normalized_mutual_info_score,
    silhouette_score,
)
from collections import Counter

# Paths
TITLES_CSV = "abstracts/data/titles_with_abstracts.csv"
SCIENTISTS_CSV = "collect_uam_data/data/scientists_with_identifiers.csv"
ORIGINAL_MODEL = "BAAI/bge-base-en-v1.5"
FINETUNED_MODEL = "ArXiv/models/bge-base-cosent-finetuned/final"

# Load data
titles_df = pd.read_csv(TITLES_CSV)
scientists_df = pd.read_csv(SCIENTISTS_CSV)
scientists_df = scientists_df[scientists_df["orcid"].isin(titles_df["main_author_orcid"])]

# Extract department (first part of affiliations)
def extract_dept(aff):
    if pd.isna(aff):
        return ""
    return str(aff).split(";")[0].strip()

scientists_df = scientists_df.copy()
scientists_df["dept"] = scientists_df["affiliations"].apply(extract_dept)

# Filter: only authors with a specific department (Zakład or Pracownia)
dept_mask = scientists_df["dept"].str.startswith("Zakład") | scientists_df["dept"].str.startswith("Pracownia")
scientists_dept = scientists_df[dept_mask].copy()

print(f"All authors: {len(scientists_df)}")
print(f"Authors with specific dept: {len(scientists_dept)}")
print(f"Departments: {scientists_dept['dept'].nunique()}")
print()
for dept, count in scientists_dept["dept"].value_counts().items():
    print(f"  [{count:2d}] {dept}")

# Build per-paper texts
texts = []
paper_orcids = []
for _, row in titles_df.iterrows():
    title = str(row["title"]) if pd.notna(row["title"]) else ""
    abstract = str(row["abstract"]) if pd.notna(row["abstract"]) else ""
    text = f"{title}. {abstract}" if abstract else title
    texts.append(text)
    paper_orcids.append(row["main_author_orcid"])

paper_orcids = np.array(paper_orcids)

# Author info (filtered to those with departments)
author_orcids = scientists_dept["orcid"].values
author_names = scientists_dept["full_name"].values
author_depts = scientists_dept["dept"].values
dept_labels = author_depts  # ground truth


def embed_and_pool(model_path, texts, paper_orcids, author_orcids):
    """Generate per-paper embeddings, then mean-pool per author."""
    print(f"\nLoading model: {model_path}")
    model = SentenceTransformer(model_path)

    print("Generating embeddings...")
    embeddings = model.encode(
        texts,
        batch_size=128,
        show_progress_bar=True,
        normalize_embeddings=True,
        convert_to_numpy=True,
    )

    author_embeddings = []
    for orcid in author_orcids:
        mask = paper_orcids == orcid
        author_emb = embeddings[mask].mean(axis=0)
        author_emb = author_emb / np.linalg.norm(author_emb)
        author_embeddings.append(author_emb)

    del model
    return np.array(author_embeddings)


def cosine_sim_matrix(emb):
    """Cosine similarity matrix (embeddings assumed normalized)."""
    return emb @ emb.T


def compute_metrics(emb, dept_labels, author_names):
    """Compute all department separation metrics."""
    n = len(dept_labels)
    unique_depts = sorted(set(dept_labels))
    n_depts = len(unique_depts)
    dept_to_idx = {d: i for i, d in enumerate(unique_depts)}

    sim_matrix = cosine_sim_matrix(emb)

    # ── Global metrics ──
    intra_sims = []
    inter_sims = []
    for i in range(n):
        for j in range(i + 1, n):
            if dept_labels[i] == dept_labels[j]:
                intra_sims.append(sim_matrix[i, j])
            else:
                inter_sims.append(sim_matrix[i, j])

    mean_intra = np.mean(intra_sims) if intra_sims else 0
    mean_inter = np.mean(inter_sims) if inter_sims else 0
    ratio = mean_intra / mean_inter if mean_inter > 0 else float("inf")

    # NN accuracy
    nn_accs = {}
    for k in [1, 3, 5]:
        correct = 0
        for i in range(n):
            sims = sim_matrix[i].copy()
            sims[i] = -1  # exclude self
            top_k = np.argsort(sims)[-k:]
            neighbor_depts = [dept_labels[j] for j in top_k]
            most_common = Counter(neighbor_depts).most_common(1)[0][0]
            if most_common == dept_labels[i]:
                correct += 1
        nn_accs[k] = correct / n

    # ── Clustering metrics ──
    kmeans = KMeans(n_clusters=n_depts, random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(emb)

    # Department purity
    purity_correct = 0
    for c in range(n_depts):
        members = [dept_labels[i] for i in range(n) if cluster_labels[i] == c]
        if members:
            most_common = Counter(members).most_common(1)[0][0]
            purity_correct += sum(1 for d in members if d == most_common)
    dept_purity = purity_correct / n

    nmi = normalized_mutual_info_score(dept_labels, cluster_labels)

    # Silhouette by true dept labels
    dept_numeric = np.array([dept_to_idx[d] for d in dept_labels])
    sil = silhouette_score(emb, dept_numeric) if n_depts > 1 else 0

    # ── Per-department breakdown ──
    dept_centroids = {}
    dept_members = {}
    for dept in unique_depts:
        mask = np.array([d == dept for d in dept_labels])
        members = emb[mask]
        centroid = members.mean(axis=0)
        centroid = centroid / np.linalg.norm(centroid)
        dept_centroids[dept] = centroid
        dept_members[dept] = members

    per_dept = {}
    for dept in unique_depts:
        members = dept_members[dept]
        n_members = len(members)

        # Intra-dept similarity
        if n_members >= 2:
            dept_sim = cosine_sim_matrix(members)
            triu = np.triu_indices(n_members, k=1)
            intra_sim = dept_sim[triu].mean()
        else:
            intra_sim = 1.0

        # Nearest other department (centroid-to-centroid)
        best_other_sim = -1
        best_other_dept = ""
        for other_dept, other_centroid in dept_centroids.items():
            if other_dept == dept:
                continue
            sim = float(dept_centroids[dept] @ other_centroid)
            if sim > best_other_sim:
                best_other_sim = sim
                best_other_dept = other_dept

        per_dept[dept] = {
            "n_members": n_members,
            "intra_sim": float(intra_sim),
            "nearest_dept": best_other_dept,
            "nearest_dept_sim": float(best_other_sim),
        }

    return {
        "mean_intra": mean_intra,
        "mean_inter": mean_inter,
        "intra_inter_ratio": ratio,
        "nn_acc": nn_accs,
        "dept_purity": dept_purity,
        "nmi": nmi,
        "silhouette": sil,
        "per_dept": per_dept,
    }


# ── Generate embeddings ──
emb_original = embed_and_pool(ORIGINAL_MODEL, texts, paper_orcids, author_orcids)
emb_finetuned = embed_and_pool(FINETUNED_MODEL, texts, paper_orcids, author_orcids)

# ── Compute metrics ──
print("\n" + "=" * 70)
print("Computing metrics...")
print("=" * 70)

m_orig = compute_metrics(emb_original, dept_labels, author_names)
m_ft = compute_metrics(emb_finetuned, dept_labels, author_names)

# ── Print results ──
print("\n" + "=" * 70)
print("GLOBAL METRICS")
print("=" * 70)
print(f"{'Metric':<35} {'Original':>10} {'Fine-tuned':>10} {'Diff':>10}")
print("-" * 70)

rows = [
    ("Mean intra-dept cos-sim", m_orig["mean_intra"], m_ft["mean_intra"]),
    ("Mean inter-dept cos-sim", m_orig["mean_inter"], m_ft["mean_inter"]),
    ("Intra/Inter ratio", m_orig["intra_inter_ratio"], m_ft["intra_inter_ratio"]),
    ("NN@1 dept accuracy", m_orig["nn_acc"][1], m_ft["nn_acc"][1]),
    ("NN@3 dept accuracy", m_orig["nn_acc"][3], m_ft["nn_acc"][3]),
    ("NN@5 dept accuracy", m_orig["nn_acc"][5], m_ft["nn_acc"][5]),
    ("Dept purity (K-means)", m_orig["dept_purity"], m_ft["dept_purity"]),
    ("NMI", m_orig["nmi"], m_ft["nmi"]),
    ("Silhouette (true depts)", m_orig["silhouette"], m_ft["silhouette"]),
]

for name, orig, ft in rows:
    diff = ft - orig
    sign = "+" if diff >= 0 else ""
    print(f"{name:<35} {orig:>10.4f} {ft:>10.4f} {sign}{diff:>9.4f}")

print("\n" + "=" * 70)
print("PER-DEPARTMENT BREAKDOWN")
print("=" * 70)

unique_depts = sorted(set(dept_labels))
print(f"\n{'Department':<55} {'N':>3} {'Intra(O)':>9} {'Intra(FT)':>9} {'Diff':>7} {'Nearest dept (FT)':>40} {'Near(O)':>8} {'Near(FT)':>8}")
print("-" * 145)

for dept in unique_depts:
    po = m_orig["per_dept"][dept]
    pf = m_ft["per_dept"][dept]
    diff_intra = pf["intra_sim"] - po["intra_sim"]
    sign = "+" if diff_intra >= 0 else ""
    print(
        f"{dept:<55} {po['n_members']:>3} "
        f"{po['intra_sim']:>9.4f} {pf['intra_sim']:>9.4f} {sign}{diff_intra:>6.4f} "
        f"{pf['nearest_dept']:>40} "
        f"{po['nearest_dept_sim']:>8.4f} {pf['nearest_dept_sim']:>8.4f}"
    )

# ── Save to wmi_report.md ──
from datetime import datetime

report_path = "ArXiv/results/finetuned_comparison/wmi_report.md"
with open(report_path, "w", encoding="utf-8") as f:
    f.write("# Ewaluacja separacji zakładów WMI w embeddingach\n\n")
    f.write(f"**Data:** {datetime.now().strftime('%Y-%m-%d %H:%M')}\n")
    f.write(f"**Model bazowy:** {ORIGINAL_MODEL}\n")
    f.write(f"**Model fine-tuned:** {FINETUNED_MODEL}\n")
    f.write(f"**Autorzy z konkretnym zakładem:** {len(author_orcids)}\n")
    f.write(f"**Zakłady/pracownie:** {len(unique_depts)}\n")
    f.write(f"**Prace (łącznie):** {len(titles_df)}\n\n")

    f.write("## Metryki globalne\n\n")
    f.write("| Metryka | Oryginalny | Fine-tuned | Diff |\n")
    f.write("|---------|-----------|------------|------|\n")
    for name, orig, ft in rows:
        diff = ft - orig
        sign = "+" if diff >= 0 else ""
        f.write(f"| {name} | {orig:.4f} | {ft:.4f} | {sign}{diff:.4f} |\n")

    f.write("\n### Objaśnienia metryk\n\n")
    f.write("- **Mean intra-dept cos-sim** — średnie cosine similarity między autorami z tego samego zakładu (wyższe = zakład bardziej spójny)\n")
    f.write("- **Mean inter-dept cos-sim** — średnie cosine similarity między autorami z różnych zakładów (niższe = lepsze rozdzielenie)\n")
    f.write("- **Intra/Inter ratio** — stosunek intra do inter (wyższy = lepsza separacja)\n")
    f.write("- **NN@k dept accuracy** — % autorów których większość z top-k najbliższych sąsiadów jest z tego samego zakładu\n")
    f.write("- **Dept purity** — K-means (k=liczba zakładów): % autorów w klastrze należących do dominującego zakładu\n")
    f.write("- **NMI** — Normalized Mutual Information między klastrami K-means a prawdziwymi zakładami (0=losowe, 1=idealne)\n")
    f.write("- **Silhouette** — jakość klastrów wg prawdziwych zakładów w pełnej 768D przestrzeni (-1 do 1)\n")

    f.write("\n## Breakdown per zakład\n\n")
    f.write("| Zakład | N | Intra (orig) | Intra (FT) | Diff | Najbliższy zakład (FT) | Near (orig) | Near (FT) |\n")
    f.write("|--------|---|-------------|-----------|------|----------------------|------------|----------|\n")
    for dept in unique_depts:
        po = m_orig["per_dept"][dept]
        pf = m_ft["per_dept"][dept]
        diff_intra = pf["intra_sim"] - po["intra_sim"]
        sign = "+" if diff_intra >= 0 else ""
        # Shorten dept name for table readability
        short_dept = dept.replace("Zakład ", "Z. ").replace("Pracownia ", "P. ")
        short_nearest = pf["nearest_dept"].replace("Zakład ", "Z. ").replace("Pracownia ", "P. ")
        f.write(
            f"| {short_dept} | {po['n_members']} "
            f"| {po['intra_sim']:.4f} | {pf['intra_sim']:.4f} | {sign}{diff_intra:.4f} "
            f"| {short_nearest} "
            f"| {po['nearest_dept_sim']:.4f} | {pf['nearest_dept_sim']:.4f} |\n"
        )

    f.write("\n### Objaśnienia kolumn\n\n")
    f.write("- **Intra** — średnie cosine similarity wewnątrz zakładu (spójność)\n")
    f.write("- **Najbliższy zakład** — zakład o najbliższym centroidzie (centroid-to-centroid cosine similarity)\n")
    f.write("- **Near** — cosine similarity do tego najbliższego zakładu (niższe = lepiej oddzielony)\n")

print(f"\nSaved: {report_path}")
