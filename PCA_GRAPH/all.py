import json
import pandas as pd
import numpy as np
import networkx as nx
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.cluster.hierarchy import dendrogram, linkage
from sentence_transformers import SentenceTransformer
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity
from pyvis.network import Network
import os

# --- 1. MODUŁ DANYCH I AI ---
def load_sara_data(file_path):
    print("1. Wczytywanie i strukturyzacja danych...")
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    authors_data = []
    corpus = []
    for author in data:
        name = author['display_name']
        titles = [w.get('title', '') for w in author.get('works', [])]
        combined_text = " ".join(titles)
        if combined_text.strip():
            stats = author.get('summary_stats', {})
            authors_data.append({
                'name': name,
                'h_index': stats.get('h_index', 0),
                'citations': author.get('cited_by_count', 0),
                'works_count': author.get('works_count', 0),
                'text': combined_text,
                'raw_works': author.get('works', [])
            })
            corpus.append(combined_text)
    return pd.DataFrame(authors_data), corpus

# --- 2. MODUŁ INTERPRETACJI PCA ---
def interpret_pca_logic(df_metadata, embeddings, pca_model):
    print("\n" + "="*50 + "\nANALIZA SEMANTYCZNA OSI PCA\n" + "="*50)
    coords = pca_model.transform(embeddings)
    for dim in range(2):
        order = coords[:, dim].argsort()
        low = df_metadata.iloc[order[:3]]['name'].tolist()
        high = df_metadata.iloc[order[-3:]]['name'].tolist()
        print(f"OS PC{dim+1}: [Minus]: {', '.join(low)} | [Plus]: {', '.join(high)}")

# --- 3. MODUŁ WIZUALIZACJI (DENDROGRAM, HEATMAP, RADAR) ---
def generate_scientific_visuals(df_metadata, embeddings, G):
    print("\n3. Generowanie wizualizacji naukowych...")

    # A. DENDROGRAM (Hierarchiczne drzewo wiedzy)
    print("   a) Tworzenie Dendrogramu...")
    linked = linkage(embeddings, method='ward')
    plt.figure(figsize=(15, 10))
    dendrogram(linked, orientation='top', labels=df_metadata['name'].tolist(),
               distance_sort='descending', show_leaf_counts=True)
    plt.title("SARA: Hierarchiczna Struktura Wydziału (Dendrogram)")
    plt.xticks(rotation=90, fontsize=8)
    plt.tight_layout()
    plt.savefig("sara_viz_dendrogram.png", dpi=300)
    plt.close()

    # B. HEATMAPA PODOBIEŃSTWA (Macierz korelacji)
    print("   b) Tworzenie Heatmapy...")
    sim_matrix = cosine_similarity(embeddings)
    df_sim = pd.DataFrame(sim_matrix, index=df_metadata['name'], columns=df_metadata['name'])
    plt.figure(figsize=(20, 18))
    sns.heatmap(df_sim, cmap='magma', vmin=0.4, vmax=1.0)
    plt.title("SARA: Macierz Podobieństwa Semantycznego")
    plt.tight_layout()
    plt.savefig("sara_viz_heatmap.png", dpi=300)
    plt.close()

def generate_expert_radar(name, df_metadata, G):
    if name not in df_metadata['name'].values: return
    info = df_metadata[df_metadata['name'] == name].iloc[0]
    metrics = {
        'H-Index': info['h_index'] / df_metadata['h_index'].max(),
        'Cytowania': np.log1p(info['citations']) / np.log1p(df_metadata['citations'].max()),
        'Liczba Prac': info['works_count'] / df_metadata['works_count'].max(),
        'Współautorzy': G.degree(name) / max(dict(G.degree()).values()) if G.has_node(name) else 0
    }
    fig = go.Figure(data=go.Scatterpolar(r=list(metrics.values()), theta=list(metrics.keys()), fill='toself'))
    fig.update_layout(polar=dict(radialaxis=dict(visible=True, range=[0, 1])), title=f"Profil: {name}", template="plotly_dark")
    fig.write_html(f"radar_{name.replace(' ', '_')}.html")

# --- 4. HYBRYDOWY MATCHING ---
def get_hybrid_matching(target_name, df_metadata, embeddings, G, w_pub=0.7, w_graph=0.3):
    idx = df_metadata[df_metadata['name'] == target_name].index[0]
    text_sims = cosine_similarity([embeddings[idx]], embeddings)[0]
    results = []
    for i, other_name in enumerate(df_metadata['name']):
        try:
            path = nx.shortest_path_length(G, target_name, other_name)
            g_sim = 1 / (1 + path)
        except: g_sim = 0
        score = (w_pub * text_sims[i]) + (w_graph * g_sim)
        results.append({'Autor': other_name, 'Score': score, 'Text': text_sims[i], 'Graph': g_sim})
    return pd.DataFrame(results).sort_values(by='Score', ascending=False).iloc[1:6]

# --- 5. GŁÓWNY POTOK ---
if __name__ == "__main__":
    df, corpus = load_sara_data('WMII_authors_with_titles.json')
    model = SentenceTransformer('allenai-specter')
    embs = model.encode(corpus, show_progress_bar=True)
    
    pca = PCA(n_components=2)
    pca.fit(embs)
    kmeans = KMeans(n_clusters=5, random_state=42, n_init=10)
    df['cluster'] = kmeans.fit_predict(embs)

    # Budowa grafu
    G = nx.Graph()
    for author in df['name']: G.add_node(author)
    work_to_authors = {}
    for _, row in df.iterrows():
        for work in row['raw_works']:
            title = work.get('title', '')
            if title:
                work_to_authors.setdefault(title, []).append(row['name'])
    for authors in work_to_authors.values():
        if len(authors) > 1:
            from itertools import combinations
            for u, v in combinations(authors, 2): G.add_edge(u, v)

    # Start analizy
    interpret_pca_logic(df, embs, pca)
    generate_scientific_visuals(df, embs, G)
    
    # Przykład dossier
    print("\nGenerowanie Dossier...")
    generate_expert_radar("Patryk Żywica", df, G)
    ranking = get_hybrid_matching("Patryk Żywica", df, embs, G)
    print(ranking.to_string(index=False))