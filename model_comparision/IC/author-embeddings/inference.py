#!/usr/bin/env python3
"""
Author Embedding Inference Script
Part of the SARA (Seek & Research) project

Usage:
    python inference.py                    # Run example
    python inference.py --find-similar "Author Name"  # Find similar authors
"""

import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


# Default model (MiniLM performed better in evaluation)
DEFAULT_MODEL = "all-MiniLM-L6-v2"

# Data paths (relative to this script)
SCRIPT_DIR = Path(__file__).parent
DATA_DIR = SCRIPT_DIR / "data"
TITLES_PATH = DATA_DIR / "titles_with_abstracts.csv"
SCIENTISTS_PATH = DATA_DIR / "scientists_with_identifiers.csv"


class AuthorEmbedder:
    """Generate and compare author embeddings based on publication texts."""

    def __init__(self, model_name: str = DEFAULT_MODEL):
        """Initialize with a sentence transformer model."""
        print(f"Loading model: {model_name}")
        self.model = SentenceTransformer(model_name)
        self.model_name = model_name
        self.author_embeddings = {}
        self.author_names = {}

    def build_author_corpus(self, titles_df: pd.DataFrame, orcid: str) -> str:
        """Build text corpus for an author by concatenating their publications."""
        works = titles_df[titles_df["main_author_orcid"] == orcid]

        chunks = []
        for _, w in works.iterrows():
            if pd.notna(w.get("title")):
                chunks.append(str(w["title"]))
            if pd.notna(w.get("abstract")):
                chunks.append(str(w["abstract"]))

        return " ".join(chunks).strip()

    def embed_author(self, text: str) -> np.ndarray:
        """Generate embedding for an author given their publication text."""
        if not text:
            raise ValueError("Empty text provided")
        return self.model.encode(text)

    def load_dataset(self, titles_path: Path = TITLES_PATH,
                     scientists_path: Path = SCIENTISTS_PATH):
        """Load and embed all authors from the dataset."""
        print(f"Loading data from:\n  {titles_path}\n  {scientists_path}")

        titles_df = pd.read_csv(titles_path)
        scientists_df = pd.read_csv(scientists_path)

        # Filter to scientists with publications
        scientists_df = scientists_df[
            scientists_df["orcid"].isin(titles_df["main_author_orcid"])
        ]

        print(f"Found {len(scientists_df)} scientists with publications")
        print("Generating embeddings...")

        for _, sci in scientists_df.iterrows():
            orcid = str(sci["orcid"])
            name = sci.get("full_name", orcid)

            text = self.build_author_corpus(titles_df, orcid)
            if text:
                self.author_embeddings[orcid] = self.embed_author(text)
                self.author_names[orcid] = name

        print(f"Embedded {len(self.author_embeddings)} authors")
        return self

    def find_similar(self, query_orcid: str = None, query_name: str = None,
                     top_k: int = 5) -> list:
        """Find most similar authors to a given author."""
        if not self.author_embeddings:
            raise ValueError("No embeddings loaded. Call load_dataset() first.")

        # Find the query author
        target_orcid = None
        if query_orcid and query_orcid in self.author_embeddings:
            target_orcid = query_orcid
        elif query_name:
            # Search by name (partial match)
            query_lower = query_name.lower()
            for orcid, name in self.author_names.items():
                if query_lower in name.lower():
                    target_orcid = orcid
                    break

        if not target_orcid:
            raise ValueError(f"Author not found: {query_orcid or query_name}")

        # Compute similarities
        target_emb = self.author_embeddings[target_orcid].reshape(1, -1)

        results = []
        for orcid, emb in self.author_embeddings.items():
            if orcid == target_orcid:
                continue
            sim = cosine_similarity(target_emb, emb.reshape(1, -1))[0][0]
            results.append({
                "orcid": orcid,
                "name": self.author_names.get(orcid, orcid),
                "similarity": float(sim)
            })

        # Sort by similarity
        results.sort(key=lambda x: x["similarity"], reverse=True)

        return results[:top_k]

    def get_embedding(self, orcid: str) -> np.ndarray:
        """Get the embedding for a specific author."""
        if orcid not in self.author_embeddings:
            raise ValueError(f"Author not found: {orcid}")
        return self.author_embeddings[orcid]


def main():
    parser = argparse.ArgumentParser(description="Author Embedding Inference")
    parser.add_argument("--model", default=DEFAULT_MODEL,
                        help="Sentence transformer model name")
    parser.add_argument("--find-similar", type=str, metavar="NAME",
                        help="Find authors similar to the given name")
    parser.add_argument("--top-k", type=int, default=5,
                        help="Number of similar authors to return")
    args = parser.parse_args()

    # Initialize embedder
    embedder = AuthorEmbedder(model_name=args.model)

    # Check if data exists
    if not TITLES_PATH.exists() or not SCIENTISTS_PATH.exists():
        print("Error: Data files not found. Expected:")
        print(f"  {TITLES_PATH}")
        print(f"  {SCIENTISTS_PATH}")
        return 1

    # Load dataset
    embedder.load_dataset()

    if args.find_similar:
        # Find similar authors
        print(f"\nFinding authors similar to: {args.find_similar}")
        print("-" * 50)

        try:
            similar = embedder.find_similar(query_name=args.find_similar,
                                           top_k=args.top_k)
            for i, author in enumerate(similar, 1):
                print(f"{i}. {author['name']}")
                print(f"   Similarity: {author['similarity']:.4f}")
                print(f"   ORCID: {author['orcid']}")
        except ValueError as e:
            print(f"Error: {e}")
            return 1
    else:
        # Demo: show a random author's similar researchers
        print("\n" + "=" * 50)
        print("DEMO: Finding similar authors")
        print("=" * 50)

        # Pick first author
        first_orcid = list(embedder.author_embeddings.keys())[0]
        first_name = embedder.author_names[first_orcid]

        print(f"\nQuery author: {first_name}")
        print("-" * 50)

        similar = embedder.find_similar(query_orcid=first_orcid, top_k=5)
        for i, author in enumerate(similar, 1):
            print(f"{i}. {author['name']}")
            print(f"   Similarity: {author['similarity']:.4f}")

        print("\n" + "-" * 50)
        print("Usage: python inference.py --find-similar \"Author Name\"")

    return 0


if __name__ == "__main__":
    exit(main())
