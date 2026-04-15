#!/usr/bin/env python3
"""
Interactive PCA visualization of WMI authors: original vs fine-tuned BGE model.

Per-paper embeddings are mean-pooled per author, then PCA 2D + K-means clustering.
Outputs interactive Plotly HTML with search bar and low-paper-count toggle.
"""

import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import json

# Paths
TITLES_CSV = "data/titles_with_abstracts.csv"
SCIENTISTS_CSV = "data/scientists_with_identifiers.csv"
ORIGINAL_MODEL = "BAAI/bge-base-en-v1.5"
FINETUNED_MODEL = "models/bge-base-cosent-finetuned/final"
N_CLUSTERS = 5
MIN_PAPERS = 5  # Threshold for "reliable" authors
OUTPUT_HTML = "results/wmi_authors/pca_authors_finetuned.html"

# Load data
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

# All author labels
orcid_to_name = dict(zip(scientists_df["orcid"], scientists_df["full_name"]))
orcid_to_dept = {}
for _, row in scientists_df.iterrows():
    aff = str(row["affiliations"]) if pd.notna(row["affiliations"]) else ""
    # Extract first part (the specific unit, e.g. "Zakład Sztucznej Inteligencji")
    dept = aff.split(";")[0].strip() if aff else ""
    orcid_to_dept[row["orcid"]] = dept

all_author_orcids = scientists_df["orcid"].values
all_author_names = [orcid_to_name[o] for o in all_author_orcids]
all_author_paper_counts = [int((paper_orcids == o).sum()) for o in all_author_orcids]
all_author_depts = [orcid_to_dept[o] for o in all_author_orcids]

# Split into reliable vs low-confidence
reliable_mask = np.array([c >= MIN_PAPERS for c in all_author_paper_counts])
low_mask = ~reliable_mask

print(f"Reliable authors (>={MIN_PAPERS} papers): {reliable_mask.sum()}")
print(f"Low-confidence authors (<{MIN_PAPERS} papers): {low_mask.sum()}")
for i in range(len(all_author_names)):
    if not reliable_mask[i]:
        print(f"  {all_author_names[i]} ({all_author_paper_counts[i]} papers)")


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


# Generate embeddings for ALL authors, both models
emb_original = embed_and_pool(ORIGINAL_MODEL, texts, paper_orcids, all_author_orcids)
emb_finetuned = embed_and_pool(FINETUNED_MODEL, texts, paper_orcids, all_author_orcids)

print(f"\nAuthor embeddings shape: {emb_original.shape}")

# PCA on ALL authors
pca_orig = PCA(n_components=2)
pca_ft = PCA(n_components=2)

coords_orig = pca_orig.fit_transform(emb_original)
coords_ft = pca_ft.fit_transform(emb_finetuned)

# K-means on reliable authors only, then predict for all
kmeans_orig = KMeans(n_clusters=N_CLUSTERS, random_state=42, n_init=10)
kmeans_ft = KMeans(n_clusters=N_CLUSTERS, random_state=42, n_init=10)

kmeans_orig.fit(coords_orig[reliable_mask])
kmeans_ft.fit(coords_ft[reliable_mask])

labels_orig = kmeans_orig.predict(coords_orig)
labels_ft = kmeans_ft.predict(coords_ft)

var_orig = sum(pca_orig.explained_variance_ratio_)
var_ft = sum(pca_ft.explained_variance_ratio_)

print(f"Variance explained - Original: {var_orig:.2%}")
print(f"Variance explained - Fine-tuned: {var_ft:.2%}")

# Build Plotly figure
fig = make_subplots(
    rows=1, cols=2,
    subplot_titles=[
        f"BGE-base (original) — variance: {var_orig:.1%}",
        f"BGE-base (CoSENT fine-tuned) — variance: {var_ft:.1%}",
    ],
    horizontal_spacing=0.08,
)

colors = ["#636efa", "#EF553B", "#00cc96", "#ab63fa", "#FFA15A",
          "#19d3f3", "#FF6692", "#B6E880", "#FF97FF", "#FECB52"]

# Track trace indices for low-confidence toggle
low_conf_trace_indices = []
trace_idx = 0

for col, (coords, labels) in enumerate([
    (coords_orig, labels_orig),
    (coords_ft, labels_ft),
], start=1):
    # Reliable authors: per-cluster traces
    for cluster_id in range(N_CLUSTERS):
        mask = (labels == cluster_id) & reliable_mask
        if not mask.any():
            trace_idx += 1
            continue
        idx = [i for i in range(len(all_author_names)) if mask[i]]
        names = [all_author_names[i] for i in idx]
        counts = [all_author_paper_counts[i] for i in idx]
        depts = [all_author_depts[i] for i in idx]

        fig.add_trace(
            go.Scatter(
                x=coords[mask, 0],
                y=coords[mask, 1],
                mode="markers",
                name=f"Cluster {cluster_id}",
                text=names,
                customdata=list(zip(names, counts, depts)),
                hovertemplate=(
                    "<b>%{customdata[0]}</b><br>"
                    "%{customdata[2]}<br>"
                    "Papers: %{customdata[1]}<br>"
                    "PC1: %{x:.3f}<br>"
                    "PC2: %{y:.3f}"
                    "<extra></extra>"
                ),
                marker=dict(size=8, color=colors[cluster_id % len(colors)]),
                legendgroup=f"cluster_{cluster_id}",
                showlegend=(col == 1),
            ),
            row=1, col=col,
        )
        trace_idx += 1

    # Low-confidence authors: single gray trace, hidden by default
    if low_mask.any():
        low_idx = [i for i in range(len(all_author_names)) if low_mask[i]]
        low_names = [all_author_names[i] for i in low_idx]
        low_counts = [all_author_paper_counts[i] for i in low_idx]
        low_depts = [all_author_depts[i] for i in low_idx]

        fig.add_trace(
            go.Scatter(
                x=coords[low_mask, 0],
                y=coords[low_mask, 1],
                mode="markers",
                name=f"<{MIN_PAPERS} papers",
                text=low_names,
                customdata=list(zip(low_names, low_counts, low_depts)),
                hovertemplate=(
                    "<b>%{customdata[0]}</b><br>"
                    "%{customdata[2]}<br>"
                    "Papers: %{customdata[1]} (low confidence)<br>"
                    "PC1: %{x:.3f}<br>"
                    "PC2: %{y:.3f}"
                    "<extra></extra>"
                ),
                marker=dict(
                    size=6,
                    color="#666666",
                    symbol="diamond",
                    line=dict(width=1, color="#999"),
                ),
                legendgroup="low_conf",
                showlegend=(col == 1),
                visible=False,
            ),
            row=1, col=col,
        )
        low_conf_trace_indices.append(trace_idx)
        trace_idx += 1

fig.update_layout(
    template="plotly_dark",
    height=700,
    width=1400,
    title=dict(
        text="PCA of WMI Authors — Mean-pooled Paper Embeddings (original vs fine-tuned)",
        x=0.5,
    ),
    legend=dict(orientation="h", yanchor="bottom", y=-0.15, xanchor="center", x=0.5),
)

fig.update_xaxes(title_text="PC1", row=1, col=1)
fig.update_xaxes(title_text="PC1", row=1, col=2)
fig.update_yaxes(title_text="PC2", row=1, col=1)
fig.update_yaxes(title_text="PC2", row=1, col=2)

# Export HTML
html_content = fig.to_html(include_plotlyjs="cdn", full_html=True)

# Inject search bar + checkbox
all_scientists_json = json.dumps(sorted(all_author_names), ensure_ascii=False)
low_conf_indices_json = json.dumps(low_conf_trace_indices)

search_bar_html = f"""
<div style="position: fixed; top: 10px; left: 50%; transform: translateX(-50%); z-index: 1000; background: rgba(0,0,0,0.85); padding: 10px 15px; border-radius: 8px; display: flex; align-items: center; gap: 10px;">
    <input type="text" id="scientistSearch" placeholder="Search scientist..."
           style="padding: 6px 10px; width: 260px; font-size: 14px; border: 1px solid #666; background: #222; color: white; border-radius: 4px;">
    <button onclick="resetHighlight()" style="padding: 6px 12px; background: #444; color: white; border: 1px solid #666; border-radius: 4px; cursor: pointer; font-size: 13px;">Reset</button>
    <label style="color: #ccc; font-size: 13px; display: flex; align-items: center; gap: 5px; cursor: pointer; white-space: nowrap;">
        <input type="checkbox" id="showLowConf" onchange="toggleLowConf()"
               style="cursor: pointer; width: 15px; height: 15px;">
        Show authors with &lt;{MIN_PAPERS} papers
    </label>
    <div id="suggestions" style="position: absolute; top: 100%; left: 15px; width: 260px; margin-top: 4px; max-height: 200px; overflow-y: auto; background: #1a1a1a; border: 1px solid #666; border-radius: 4px; display: none;"></div>
</div>

<script>
var scientists = {all_scientists_json};
var lowConfTraceIndices = {low_conf_indices_json};

var searchInput = document.getElementById('scientistSearch');
var suggestionsDiv = document.getElementById('suggestions');

searchInput.addEventListener('input', function() {{
    var query = this.value.toLowerCase();
    suggestionsDiv.innerHTML = '';

    if (query.length === 0) {{
        suggestionsDiv.style.display = 'none';
        return;
    }}

    var matches = scientists.filter(function(name) {{
        return name.toLowerCase().includes(query);
    }});

    if (matches.length > 0) {{
        suggestionsDiv.style.display = 'block';
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
                highlightScientist(name);
                searchInput.value = name;
                suggestionsDiv.style.display = 'none';
            }};
            suggestionsDiv.appendChild(div);
        }});
    }} else {{
        suggestionsDiv.style.display = 'none';
    }}
}});

function toggleLowConf() {{
    var gd = document.getElementsByClassName('plotly-graph-div')[0];
    var show = document.getElementById('showLowConf').checked;
    var update = {{}};
    for (var i = 0; i < lowConfTraceIndices.length; i++) {{
        Plotly.restyle(gd, {{'visible': show}}, [lowConfTraceIndices[i]]);
    }}
}}

function highlightScientist(scientistName) {{
    var gd = document.getElementsByClassName('plotly-graph-div')[0];
    // Make sure low-conf traces are visible if searching for one
    var checkbox = document.getElementById('showLowConf');

    var update = {{
        'marker.size': [],
        'marker.line.width': [],
        'marker.line.color': [],
        'marker.opacity': []
    }};

    var foundInLowConf = false;

    for (var i = 0; i < gd.data.length; i++) {{
        var trace = gd.data[i];
        var sizes = [];
        var lineWidths = [];
        var lineColors = [];
        var opacities = [];

        for (var j = 0; j < trace.text.length; j++) {{
            if (trace.text[j] === scientistName) {{
                sizes.push(22);
                lineWidths.push(3);
                lineColors.push('yellow');
                opacities.push(1.0);
                if (lowConfTraceIndices.indexOf(i) !== -1) {{
                    foundInLowConf = true;
                }}
            }} else {{
                sizes.push(6);
                lineWidths.push(0);
                lineColors.push('');
                opacities.push(0.3);
            }}
        }}

        update['marker.size'].push(sizes);
        update['marker.line.width'].push(lineWidths);
        update['marker.line.color'].push(lineColors);
        update['marker.opacity'].push(opacities);
    }}

    // Auto-show low-conf traces if searching for a low-conf author
    if (foundInLowConf && !checkbox.checked) {{
        checkbox.checked = true;
        toggleLowConf();
    }}

    Plotly.restyle(gd, update);
}}

function resetHighlight() {{
    var gd = document.getElementsByClassName('plotly-graph-div')[0];
    Plotly.restyle(gd, {{
        'marker.size': 8,
        'marker.line.width': 0,
        'marker.line.color': '',
        'marker.opacity': 1.0
    }});
    searchInput.value = '';
    suggestionsDiv.style.display = 'none';
}}

document.addEventListener('click', function(event) {{
    if (!searchInput.contains(event.target) && !suggestionsDiv.contains(event.target)) {{
        suggestionsDiv.style.display = 'none';
    }}
}});
</script>
"""

html_content = html_content.replace("</body>", search_bar_html + "\n</body>")

with open(OUTPUT_HTML, "w", encoding="utf-8") as f:
    f.write(html_content)

print(f"\nSaved: {OUTPUT_HTML}")
