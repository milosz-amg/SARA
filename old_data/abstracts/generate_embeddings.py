#!/usr/bin/env python3
"""
Generate embeddings for titles_with_abstracts.csv using fine-tuned BGE model.
Outputs: embeddings.npy + metadata.csv
"""

import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer

CSV_PATH = "data/titles_with_abstracts.csv"
MODEL_PATH = "/home/jakub/Projekty/SARA/ArXiv/final-bge-finetuned"
OUTPUT_EMBEDDINGS = "data/embeddings.npy"
OUTPUT_METADATA = "data/embeddings_metadata.csv"

# Load papers
df = pd.read_csv(CSV_PATH)
print(f"Loaded {len(df)} papers")

# Combine title + abstract
texts = []
for _, row in df.iterrows():
    title = str(row['title']) if pd.notna(row['title']) else ""
    abstract = str(row['abstract']) if pd.notna(row['abstract']) else ""
    if abstract:
        texts.append(f"{title}. {abstract}")
    else:
        texts.append(title)

print(f"Texts prepared: {len(texts)}")
print(f"Sample: {texts[0][:100]}...")

# Load model
print(f"\nLoading model: {MODEL_PATH}")
model = SentenceTransformer(MODEL_PATH)
print(f"Embedding dimension: {model.get_sentence_embedding_dimension()}")

# Generate embeddings
print("\nGenerating embeddings...")
embeddings = model.encode(
    texts,
    batch_size=128,
    show_progress_bar=True,
    normalize_embeddings=True,
    convert_to_numpy=True
)

print(f"Embeddings shape: {embeddings.shape}")

# Save embeddings
np.save(OUTPUT_EMBEDDINGS, embeddings)
print(f"Saved embeddings to: {OUTPUT_EMBEDDINGS}")

# Save metadata (maps row index to paper info)
meta = df[['openalex_id', 'title', 'main_author_orcid', 'publication_year']].copy()
meta.to_csv(OUTPUT_METADATA, index=True)
print(f"Saved metadata to: {OUTPUT_METADATA}")

print(f"\nDone! {embeddings.shape[0]} papers x {embeddings.shape[1]} dimensions")
