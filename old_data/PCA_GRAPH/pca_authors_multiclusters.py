#!/usr/bin/env python3
"""
Multi-cluster PCA visualization of WMI authors.

Each author is represented as multiple points — one per research area cluster.
Papers are clustered per-author using K-means on 768D embeddings, with automatic
k selection via silhouette score. Output: interactive Plotly HTML.
"""

import os
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from collections import Counter
import plotly.graph_objects as go
import json

# ── Config ──
TITLES_CSV = "abstracts/data/titles_with_abstracts.csv"
SCIENTISTS_CSV = "collect_uam_data/data/scientists_with_identifiers.csv"
FINETUNED_MODEL = "ArXiv/models/bge-base-cosent-finetuned/final"
CACHE_PATH = "PCA_GRAPH/paper_embeddings_cosent.npy"
OUTPUT_HTML = "PCA_GRAPH/pca_authors_multiclusters.html"

MIN_PAPERS_FOR_CLUSTERING = 5
MAX_CLUSTERS = 5
SILHOUETTE_THRESHOLD = 0.15

# ── Load data ──
titles_df = pd.read_csv(TITLES_CSV)
scientists_df = pd.read_csv(SCIENTISTS_CSV)
scientists_df = scientists_df[scientists_df["orcid"].isin(titles_df["main_author_orcid"])]

print(f"Scientists: {len(scientists_df)}, Papers: {len(titles_df)}")

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

# Author metadata
orcid_to_name = dict(zip(scientists_df["orcid"], scientists_df["full_name"]))
orcid_to_dept = {}
for _, row in scientists_df.iterrows():
    aff = str(row["affiliations"]) if pd.notna(row["affiliations"]) else ""
    dept = aff.split(";")[0].strip() if aff else ""
    orcid_to_dept[row["orcid"]] = dept

all_author_orcids = scientists_df["orcid"].values

# ── Generate or load embeddings ──
if os.path.exists(CACHE_PATH):
    print(f"\nLoading cached embeddings: {CACHE_PATH}")
    all_embeddings = np.load(CACHE_PATH)
    print(f"Shape: {all_embeddings.shape}")
else:
    print(f"\nLoading model: {FINETUNED_MODEL}")
    model = SentenceTransformer(FINETUNED_MODEL)
    print("Generating embeddings...")
    all_embeddings = model.encode(
        texts,
        batch_size=128,
        show_progress_bar=True,
        normalize_embeddings=True,
        convert_to_numpy=True,
    )
    del model
    np.save(CACHE_PATH, all_embeddings)
    print(f"Saved cache: {CACHE_PATH}")


# ── Per-author sub-clustering ──
def cluster_author_papers(author_embeddings):
    """Cluster an author's papers into research areas. Returns (k, labels, centroids)."""
    n = len(author_embeddings)

    if n <= 1:
        return 1, np.array([0]), author_embeddings.copy()

    if n < MIN_PAPERS_FOR_CLUSTERING:
        centroid = author_embeddings.mean(axis=0, keepdims=True)
        centroid = centroid / np.linalg.norm(centroid, axis=1, keepdims=True)
        return 1, np.zeros(n, dtype=int), centroid

    max_k = min(MAX_CLUSTERS, n - 1)
    if max_k < 2:
        centroid = author_embeddings.mean(axis=0, keepdims=True)
        centroid = centroid / np.linalg.norm(centroid, axis=1, keepdims=True)
        return 1, np.zeros(n, dtype=int), centroid

    best_k = 1
    best_sil = -1
    best_labels = None

    for k in range(2, max_k + 1):
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = km.fit_predict(author_embeddings)
        if len(set(labels)) < 2:
            continue
        sil = silhouette_score(author_embeddings, labels)
        if sil > best_sil:
            best_sil = sil
            best_k = k
            best_labels = labels

    if best_sil < SILHOUETTE_THRESHOLD or best_labels is None:
        centroid = author_embeddings.mean(axis=0, keepdims=True)
        centroid = centroid / np.linalg.norm(centroid, axis=1, keepdims=True)
        return 1, np.zeros(n, dtype=int), centroid

    centroids = []
    for c in range(best_k):
        mask = best_labels == c
        centroid = author_embeddings[mask].mean(axis=0)
        centroid = centroid / np.linalg.norm(centroid)
        centroids.append(centroid)

    return best_k, best_labels, np.array(centroids)


def get_cluster_label(paper_indices):
    """Extract top topics and keywords for papers in a cluster."""
    all_topics = []
    all_kw = []
    for idx in paper_indices:
        row = titles_df.iloc[idx]
        if pd.notna(row.get("topics")):
            all_topics.extend([t.strip() for t in str(row["topics"]).split(";")])
        if pd.notna(row.get("keywords")):
            all_kw.extend([k.strip() for k in str(row["keywords"]).split(";")])

    top_topics = [t for t, _ in Counter(all_topics).most_common(3)]
    top_kw = [k for k, _ in Counter(all_kw).most_common(5)]
    return top_topics, top_kw


# ── Build cluster points ──
print("\nClustering authors...")
cluster_points = []

for orcid in all_author_orcids:
    mask = paper_orcids == orcid
    paper_indices = np.where(mask)[0]
    author_embs = all_embeddings[mask]
    name = orcid_to_name[orcid]
    dept = orcid_to_dept[orcid]
    n_papers = len(author_embs)

    k, labels, centroids = cluster_author_papers(author_embs)

    for c in range(k):
        cluster_paper_idx = paper_indices[labels == c]
        top_topics, top_kw = get_cluster_label(cluster_paper_idx)

        cluster_points.append({
            "orcid": orcid,
            "name": name,
            "dept": dept,
            "total_papers": n_papers,
            "cluster_id": c,
            "n_clusters": k,
            "cluster_papers": len(cluster_paper_idx),
            "centroid": centroids[c],
            "top_topics": top_topics,
            "top_keywords": top_kw,
        })

n_multi = sum(1 for o in all_author_orcids
              if sum(1 for cp in cluster_points if cp["orcid"] == o) > 1)
avg_k = len(cluster_points) / len(all_author_orcids)

print(f"Total cluster-points: {len(cluster_points)}")
print(f"Authors with >1 cluster: {n_multi}")
print(f"Avg clusters/author: {avg_k:.1f}")

# Print multi-cluster authors
print("\nMulti-cluster authors:")
for orcid in all_author_orcids:
    pts = [cp for cp in cluster_points if cp["orcid"] == orcid]
    if len(pts) > 1:
        name = orcid_to_name[orcid]
        print(f"  {name}: {len(pts)} clusters")
        for cp in pts:
            topics_str = "; ".join(cp["top_topics"][:2]) if cp["top_topics"] else "?"
            print(f"    Cluster {cp['cluster_id']+1}: {cp['cluster_papers']} papers — {topics_str}")

# ── PCA ──
all_centroids = np.array([cp["centroid"] for cp in cluster_points])
pca = PCA(n_components=2)
coords_2d = pca.fit_transform(all_centroids)
var_explained = sum(pca.explained_variance_ratio_)

for i, cp in enumerate(cluster_points):
    cp["x"] = float(coords_2d[i, 0])
    cp["y"] = float(coords_2d[i, 1])

print(f"\nVariance explained: {var_explained:.1%}")

# ── Build Plotly figure ──
fig = go.Figure()

# Assign colors by department
unique_depts = sorted(set(cp["dept"] for cp in cluster_points))
color_palette = [
    "#636efa", "#EF553B", "#00cc96", "#ab63fa", "#FFA15A",
    "#19d3f3", "#FF6692", "#B6E880", "#FF97FF", "#FECB52",
    "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
    "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22",
]
dept_colors = {d: color_palette[i % len(color_palette)] for i, d in enumerate(unique_depts)}

# Scatter traces per department
for dept in unique_depts:
    dept_pts = [cp for cp in cluster_points if cp["dept"] == dept]

    xs = [cp["x"] for cp in dept_pts]
    ys = [cp["y"] for cp in dept_pts]
    sizes = [max(6, min(18, 4 + 2 * np.sqrt(cp["cluster_papers"]))) for cp in dept_pts]
    texts = [cp["name"] for cp in dept_pts]

    customdata = []
    for cp in dept_pts:
        label = f"({cp['cluster_id']+1}/{cp['n_clusters']})" if cp["n_clusters"] > 1 else ""
        topics_str = "; ".join(cp["top_topics"][:3])
        kw_str = "; ".join(cp["top_keywords"][:5])
        customdata.append([
            cp["name"], label, cp["dept"],
            cp["cluster_papers"], cp["total_papers"],
            topics_str, kw_str,
        ])

    short_dept = dept.replace("Zakład ", "Z. ").replace("Pracownia ", "P. ")
    if len(short_dept) > 50:
        short_dept = short_dept[:47] + "..."

    fig.add_trace(go.Scatter(
        x=xs, y=ys,
        mode="markers",
        name=short_dept,
        text=texts,
        customdata=customdata,
        hovertemplate=(
            "<b>%{customdata[0]}</b> %{customdata[1]}<br>"
            "%{customdata[2]}<br>"
            "Papers: %{customdata[3]} / %{customdata[4]} total<br>"
            "Topics: %{customdata[5]}<br>"
            "Keywords: %{customdata[6]}"
            "<extra></extra>"
        ),
        marker=dict(
            size=sizes,
            color=dept_colors[dept],
            line=dict(width=0.5, color="rgba(255,255,255,0.3)"),
        ),
        legendgroup=dept,
    ))

# Lines connecting clusters of the same author
line_x = []
line_y = []
for orcid in all_author_orcids:
    pts = [(cp["x"], cp["y"]) for cp in cluster_points if cp["orcid"] == orcid]
    if len(pts) > 1:
        for i in range(len(pts)):
            for j in range(i + 1, len(pts)):
                line_x.extend([pts[i][0], pts[j][0], None])
                line_y.extend([pts[i][1], pts[j][1], None])

fig.add_trace(go.Scatter(
    x=line_x, y=line_y,
    mode="lines",
    name="Author cluster links",
    line=dict(color="rgba(255,255,255,0.2)", width=1),
    hoverinfo="skip",
    showlegend=True,
    visible=True,
))
line_trace_idx = len(fig.data) - 1

fig.update_layout(
    template="plotly_dark",
    height=800,
    width=1200,
    title=dict(
        text=(
            f"PCA of WMI Authors — Multi-Cluster View (BGE CoSENT fine-tuned)<br>"
            f"<sup>{len(all_author_orcids)} authors, {len(cluster_points)} cluster-points, "
            f"avg {avg_k:.1f} clusters/author, "
            f"variance explained: {var_explained:.1%}</sup>"
        ),
        x=0.5,
    ),
    legend=dict(
        orientation="v",
        yanchor="top", y=0.98,
        xanchor="left", x=1.02,
        font=dict(size=9),
    ),
    xaxis=dict(title="PC1"),
    yaxis=dict(title="PC2"),
    margin=dict(r=250),
)

# ── Export HTML with search bar + toggles ──
html_content = fig.to_html(include_plotlyjs="cdn", full_html=True)

all_scientists = sorted(set(cp["name"] for cp in cluster_points))
scientists_json = json.dumps(all_scientists, ensure_ascii=False)

search_bar_html = f"""
<div style="position: fixed; top: 10px; left: 50%; transform: translateX(-50%); z-index: 1000; background: rgba(0,0,0,0.85); padding: 10px 15px; border-radius: 8px; display: flex; align-items: center; gap: 10px; flex-wrap: nowrap;">
    <div style="position: relative;">
        <input type="text" id="searchA" placeholder="Scientist A..."
               style="padding: 6px 10px; width: 200px; font-size: 14px; border: 2px solid #f0e130; background: #222; color: white; border-radius: 4px;">
        <div id="suggestionsA" style="position: absolute; top: 100%; left: 0; width: 200px; margin-top: 4px; max-height: 200px; overflow-y: auto; background: #1a1a1a; border: 1px solid #666; border-radius: 4px; display: none; z-index: 1001;"></div>
    </div>
    <span style="color: #888; font-size: 14px;">vs</span>
    <div style="position: relative;">
        <input type="text" id="searchB" placeholder="Scientist B..."
               style="padding: 6px 10px; width: 200px; font-size: 14px; border: 2px solid #00e5ff; background: #222; color: white; border-radius: 4px;">
        <div id="suggestionsB" style="position: absolute; top: 100%; left: 0; width: 200px; margin-top: 4px; max-height: 200px; overflow-y: auto; background: #1a1a1a; border: 1px solid #666; border-radius: 4px; display: none; z-index: 1001;"></div>
    </div>
    <button onclick="resetHighlight()" style="padding: 6px 12px; background: #444; color: white; border: 1px solid #666; border-radius: 4px; cursor: pointer; font-size: 13px;">Reset</button>
    <label style="color: #ccc; font-size: 13px; display: flex; align-items: center; gap: 5px; cursor: pointer; white-space: nowrap;">
        <input type="checkbox" id="showLines" checked onchange="toggleLines()"
               style="cursor: pointer; width: 15px; height: 15px;">
        Cluster links
    </label>
</div>

<script>
var scientists = {scientists_json};
var lineTraceIdx = {line_trace_idx};

var selectedA = null;
var selectedB = null;

function setupSearch(inputId, suggestionsId, slot) {{
    var input = document.getElementById(inputId);
    var sugDiv = document.getElementById(suggestionsId);

    input.addEventListener('input', function() {{
        var query = this.value.toLowerCase();
        sugDiv.innerHTML = '';

        if (query.length === 0) {{
            sugDiv.style.display = 'none';
            if (slot === 'A') selectedA = null;
            else selectedB = null;
            applyHighlight();
            return;
        }}

        var matches = scientists.filter(function(name) {{
            return name.toLowerCase().includes(query);
        }});

        if (matches.length > 0) {{
            sugDiv.style.display = 'block';
            matches.forEach(function(name) {{
                var div = document.createElement('div');
                div.textContent = name;
                div.style.padding = '5px 10px';
                div.style.cursor = 'pointer';
                div.style.color = 'white';
                div.style.fontSize = '13px';
                div.onmouseover = function() {{ this.style.background = '#444'; }};
                div.onmouseout = function() {{ this.style.background = 'transparent'; }};
                div.onclick = function() {{
                    input.value = name;
                    sugDiv.style.display = 'none';
                    if (slot === 'A') selectedA = name;
                    else selectedB = name;
                    applyHighlight();
                }};
                sugDiv.appendChild(div);
            }});
        }} else {{
            sugDiv.style.display = 'none';
        }}
    }});
}}

setupSearch('searchA', 'suggestionsA', 'A');
setupSearch('searchB', 'suggestionsB', 'B');

function toggleLines() {{
    var gd = document.getElementsByClassName('plotly-graph-div')[0];
    var show = document.getElementById('showLines').checked;
    Plotly.restyle(gd, {{'visible': show}}, [lineTraceIdx]);
}}

function applyHighlight() {{
    var gd = document.getElementsByClassName('plotly-graph-div')[0];
    var anySelected = (selectedA !== null || selectedB !== null);

    var update = {{
        'marker.size': [],
        'marker.line.width': [],
        'marker.line.color': [],
        'marker.opacity': []
    }};

    for (var i = 0; i < gd.data.length; i++) {{
        var trace = gd.data[i];
        if (!trace.text || trace.mode === 'lines') {{
            update['marker.size'].push(undefined);
            update['marker.line.width'].push(undefined);
            update['marker.line.color'].push(undefined);
            update['marker.opacity'].push(undefined);
            continue;
        }}

        var sizes = [];
        var lineWidths = [];
        var lineColors = [];
        var opacities = [];

        for (var j = 0; j < trace.text.length; j++) {{
            var name = trace.text[j];
            var baseSize = trace.marker.size[j] || 8;

            if (name === selectedA) {{
                sizes.push(22);
                lineWidths.push(3);
                lineColors.push('#f0e130');
                opacities.push(1.0);
            }} else if (name === selectedB) {{
                sizes.push(22);
                lineWidths.push(3);
                lineColors.push('#00e5ff');
                opacities.push(1.0);
            }} else {{
                sizes.push(baseSize);
                lineWidths.push(anySelected ? 0 : 0.5);
                lineColors.push(anySelected ? '' : 'rgba(255,255,255,0.3)');
                opacities.push(anySelected ? 0.15 : 1.0);
            }}
        }}

        update['marker.size'].push(sizes);
        update['marker.line.width'].push(lineWidths);
        update['marker.line.color'].push(lineColors);
        update['marker.opacity'].push(opacities);
    }}

    Plotly.restyle(gd, update);
}}

function resetHighlight() {{
    selectedA = null;
    selectedB = null;
    document.getElementById('searchA').value = '';
    document.getElementById('searchB').value = '';
    document.getElementById('suggestionsA').style.display = 'none';
    document.getElementById('suggestionsB').style.display = 'none';

    var gd = document.getElementsByClassName('plotly-graph-div')[0];
    for (var i = 0; i < gd.data.length; i++) {{
        var trace = gd.data[i];
        if (!trace.text || trace.mode === 'lines') continue;

        Plotly.restyle(gd, {{
            'marker.line.width': 0.5,
            'marker.line.color': 'rgba(255,255,255,0.3)',
            'marker.opacity': 1.0,
        }}, [i]);
    }}
}}

document.addEventListener('click', function(event) {{
    var sugA = document.getElementById('suggestionsA');
    var sugB = document.getElementById('suggestionsB');
    var inA = document.getElementById('searchA');
    var inB = document.getElementById('searchB');
    if (!inA.contains(event.target) && !sugA.contains(event.target)) {{
        sugA.style.display = 'none';
    }}
    if (!inB.contains(event.target) && !sugB.contains(event.target)) {{
        sugB.style.display = 'none';
    }}
}});
</script>
"""

html_content = html_content.replace("</body>", search_bar_html + "\n</body>")

with open(OUTPUT_HTML, "w", encoding="utf-8") as f:
    f.write(html_content)

print(f"\nSaved: {OUTPUT_HTML}")
