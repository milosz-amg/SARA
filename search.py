import faiss
import json
import numpy as np
from embedder import embed_text

def search_faiss(query, index_path, top_k=3):
    index = faiss.read_index(index_path)
    with open(index_path + ".meta.json", encoding="utf-8") as f:
        metadata = json.load(f)

    query_embedding = np.array([embed_text(query)]).astype("float32")
    D, I = index.search(query_embedding, top_k)

    results = [metadata[i] for i in I[0]]
    return results
