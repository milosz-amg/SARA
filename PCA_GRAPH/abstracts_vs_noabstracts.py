import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sentence_transformers import SentenceTransformer
from sklearn.decomposition import PCA

# -----------------------------
# 1. LOAD DATA
# -----------------------------
titles_df = pd.read_csv("abstracts/data/titles_with_abstracts.csv")
scientists_df = pd.read_csv("collect_uam_data/data/scientists_with_identifiers.csv")

# Keep only scientists that actually appear in works
scientists_df = scientists_df[
    scientists_df["orcid"].isin(titles_df["main_author_orcid"])
]

# -----------------------------
# 2. BUILD AUTHOR CORPORA
# -----------------------------
def build_author_corpora(titles_df, scientists_df, use_abstracts: bool):
    texts = []
    labels = []

    for _, sci in scientists_df.iterrows():
        orcid = sci["orcid"]
        name = sci["full_name"]

        works = titles_df[titles_df["main_author_orcid"] == orcid]

        chunks = []
        for _, w in works.iterrows():
            title = str(w["title"]) if pd.notna(w["title"]) else ""
            chunks.append(title)

            if use_abstracts:
                abstract = str(w.get("abstract", "")) if pd.notna(w.get("abstract")) else ""
                chunks.append(abstract)

        combined_text = " ".join(chunks).strip()

        if combined_text:
            texts.append(combined_text)
            labels.append(name)

    return texts, labels


texts_titles, labels = build_author_corpora(
    titles_df, scientists_df, use_abstracts=False
)

texts_titles_abstracts, _ = build_author_corpora(
    titles_df, scientists_df, use_abstracts=True
)

print(f"Scientists used: {len(labels)}")

# -----------------------------
# 3. EMBEDDINGS
# -----------------------------
model = SentenceTransformer("allenai-specter")

emb_titles = model.encode(texts_titles, show_progress_bar=True)
emb_titles_abs = model.encode(texts_titles_abstracts, show_progress_bar=True)

# -----------------------------
# 4. PCA
# -----------------------------
pca = PCA(n_components=2)

coords_titles = pca.fit_transform(emb_titles)
coords_titles_abs = pca.fit_transform(emb_titles_abs)

# -----------------------------
# 5. VISUALIZATION
# -----------------------------
fig, axes = plt.subplots(1, 2, figsize=(16, 7), sharex=False, sharey=False)

axes[0].scatter(coords_titles[:, 0], coords_titles[:, 1], alpha=0.7)
axes[0].set_title("PCA — Titles only")

axes[1].scatter(coords_titles_abs[:, 0], coords_titles_abs[:, 1], alpha=0.7)
axes[1].set_title("PCA — Titles + Abstracts")

for ax in axes:
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")

plt.tight_layout()
plt.show()


# -----------------------------
# 6. DISPLACEMENT PLOT  ⬅️ ADD HERE
# -----------------------------
plt.figure(figsize=(8, 8))

for i in range(len(labels)):
    x0, y0 = coords_titles[i]
    x1, y1 = coords_titles_abs[i]

    plt.plot([x0, x1], [y0, y1], alpha=0.4)
    plt.scatter(x0, y0)
    plt.scatter(x1, y1, marker="x")

plt.title("Scientist displacement: Titles → Titles+Abstracts")
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.show()