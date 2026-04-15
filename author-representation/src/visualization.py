"""
Visualization Utilities

Plotly-based visualization helpers for PCA plots and cluster analysis.
Follows patterns from PCA_GRAPH/all.py
"""

import os
import json
import logging
from typing import List, Dict, Optional, Tuple
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

logger = logging.getLogger(__name__)


def plot_pca_2d(
    coords: np.ndarray,
    labels: np.ndarray,
    metadata: Dict,
    title: str,
    output_path: str,
    color_by: str = 'category',
    hover_data: Optional[Dict] = None
) -> go.Figure:
    """
    Create interactive 2D PCA scatter plot.

    Args:
        coords: PCA coordinates (N, 2)
        labels: Category or cluster labels for coloring
        metadata: Dict with 'titles', 'ids', etc.
        title: Plot title
        output_path: Path to save HTML file
        color_by: What labels represent ('category' or 'cluster')
        hover_data: Additional data for hover tooltips

    Returns:
        Plotly figure object
    """
    # Prepare data for plotting
    plot_data = {
        'PC1': coords[:, 0],
        'PC2': coords[:, 1],
        'label': labels,
        'title': metadata.get('titles', [f'Paper {i}' for i in range(len(coords))]),
        'id': metadata.get('ids', [str(i) for i in range(len(coords))])
    }

    # Add hover data if provided
    if hover_data:
        plot_data.update(hover_data)

    # Create figure
    fig = px.scatter(
        plot_data,
        x='PC1',
        y='PC2',
        color='label',
        hover_data=['id', 'title'],
        title=title,
        labels={
            'PC1': f'PC1 ({metadata.get("variance_pc1", 0):.1%} variance)',
            'PC2': f'PC2 ({metadata.get("variance_pc2", 0):.1%} variance)',
            'label': color_by.capitalize()
        }
    )

    # Update layout
    fig.update_layout(
        width=1200,
        height=800,
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=1.02
        ),
        hovermode='closest'
    )

    # Update marker size
    fig.update_traces(marker=dict(size=6, opacity=0.7))

    # Save to HTML
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig.write_html(output_path)
    logger.info(f"Saved 2D plot to {output_path}")

    return fig


def plot_pca_3d(
    coords: np.ndarray,
    labels: np.ndarray,
    metadata: Dict,
    title: str,
    output_path: str,
    color_by: str = 'category'
) -> go.Figure:
    """
    Create interactive 3D PCA scatter plot.

    Args:
        coords: PCA coordinates (N, 3)
        labels: Category or cluster labels for coloring
        metadata: Dict with 'titles', 'ids', etc.
        title: Plot title
        output_path: Path to save HTML file
        color_by: What labels represent ('category' or 'cluster')

    Returns:
        Plotly figure object
    """
    # Create figure
    fig = go.Figure(data=[go.Scatter3d(
        x=coords[:, 0],
        y=coords[:, 1],
        z=coords[:, 2],
        mode='markers',
        marker=dict(
            size=4,
            color=labels if isinstance(labels[0], (int, float)) else None,
            colorscale='Viridis',
            opacity=0.7
        ),
        text=[f"ID: {id}<br>Title: {t[:50]}..."
              for id, t in zip(metadata.get('ids', []), metadata.get('titles', []))],
        hoverinfo='text'
    )])

    # If labels are categorical, use different approach
    if not isinstance(labels[0], (int, float)) or len(np.unique(labels)) < 50:
        unique_labels = np.unique(labels)
        fig = go.Figure()

        for label in unique_labels:
            mask = labels == label
            fig.add_trace(go.Scatter3d(
                x=coords[mask, 0],
                y=coords[mask, 1],
                z=coords[mask, 2],
                mode='markers',
                name=str(label),
                marker=dict(size=4, opacity=0.7),
                text=[f"ID: {id}<br>Title: {t[:50]}...<br>{color_by}: {label}"
                      for id, t in zip(
                          np.array(metadata.get('ids', []))[mask],
                          np.array(metadata.get('titles', []))[mask]
                      )],
                hoverinfo='text'
            ))

    # Update layout
    fig.update_layout(
        title=title,
        width=1200,
        height=900,
        scene=dict(
            xaxis_title=f'PC1 ({metadata.get("variance_pc1", 0):.1%})',
            yaxis_title=f'PC2 ({metadata.get("variance_pc2", 0):.1%})',
            zaxis_title=f'PC3 ({metadata.get("variance_pc3", 0):.1%})'
        ),
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=1.02
        )
    )

    # Save to HTML
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig.write_html(output_path)
    logger.info(f"Saved 3D plot to {output_path}")

    return fig


def plot_elbow_curve(
    k_values: List[int],
    inertias: List[float],
    silhouettes: List[float],
    optimal_k: int,
    title: str,
    output_path: str
) -> go.Figure:
    """
    Plot elbow curve for K-means clustering.

    Args:
        k_values: List of k values tested
        inertias: Inertia (within-cluster sum of squares) for each k
        silhouettes: Silhouette scores for each k
        optimal_k: The selected optimal k
        title: Plot title
        output_path: Path to save HTML file

    Returns:
        Plotly figure object
    """
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=('Elbow Method (Inertia)', 'Silhouette Score')
    )

    # Elbow curve
    fig.add_trace(
        go.Scatter(x=k_values, y=inertias, mode='lines+markers', name='Inertia'),
        row=1, col=1
    )
    fig.add_vline(x=optimal_k, line_dash="dash", line_color="red", row=1, col=1)

    # Silhouette score
    fig.add_trace(
        go.Scatter(x=k_values, y=silhouettes, mode='lines+markers', name='Silhouette'),
        row=1, col=2
    )
    fig.add_vline(x=optimal_k, line_dash="dash", line_color="red", row=1, col=2)

    fig.update_layout(
        title=title,
        width=1200,
        height=500,
        showlegend=False
    )

    fig.update_xaxes(title_text="Number of Clusters (k)", row=1, col=1)
    fig.update_xaxes(title_text="Number of Clusters (k)", row=1, col=2)
    fig.update_yaxes(title_text="Inertia", row=1, col=1)
    fig.update_yaxes(title_text="Silhouette Score", row=1, col=2)

    # Save
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig.write_html(output_path)
    logger.info(f"Saved elbow curve to {output_path}")

    return fig


def plot_category_distribution(
    cluster_assignments: np.ndarray,
    categories: np.ndarray,
    title: str,
    output_path: str
) -> go.Figure:
    """
    Plot category distribution within each cluster as a heatmap.

    Args:
        cluster_assignments: Cluster ID for each paper
        categories: Category for each paper
        title: Plot title
        output_path: Path to save HTML file

    Returns:
        Plotly figure object
    """
    unique_clusters = sorted(np.unique(cluster_assignments))
    unique_categories = sorted(np.unique(categories))

    # Build distribution matrix
    matrix = np.zeros((len(unique_clusters), len(unique_categories)))

    for i, cluster in enumerate(unique_clusters):
        cluster_mask = cluster_assignments == cluster
        cluster_categories = categories[cluster_mask]
        for j, cat in enumerate(unique_categories):
            matrix[i, j] = np.sum(cluster_categories == cat)

    # Normalize by row (cluster)
    row_sums = matrix.sum(axis=1, keepdims=True)
    matrix_normalized = np.divide(matrix, row_sums, where=row_sums != 0)

    fig = go.Figure(data=go.Heatmap(
        z=matrix_normalized,
        x=unique_categories,
        y=[f'Cluster {c}' for c in unique_clusters],
        colorscale='Blues',
        text=matrix.astype(int),
        texttemplate='%{text}',
        textfont={"size": 8},
        hovertemplate='Cluster: %{y}<br>Category: %{x}<br>Count: %{text}<br>Proportion: %{z:.2%}<extra></extra>'
    ))

    fig.update_layout(
        title=title,
        width=1400,
        height=600,
        xaxis_title='ArXiv Category',
        yaxis_title='Cluster',
        xaxis={'tickangle': 45}
    )

    # Save
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig.write_html(output_path)
    logger.info(f"Saved category distribution to {output_path}")

    return fig


def plot_model_comparison(
    results: List[Dict],
    output_path: str
) -> go.Figure:
    """
    Create comparison chart for multiple embedding models.

    Args:
        results: List of dicts with model evaluation results
        output_path: Path to save HTML file

    Returns:
        Plotly figure object
    """
    model_names = [r['model_name'].split('/')[-1] for r in results]
    purities = [r.get('category_purity', 0) for r in results]
    silhouettes = [r.get('silhouette_score', 0) for r in results]

    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=('Category Purity (%)', 'Silhouette Score')
    )

    # Sort by purity
    sorted_indices = np.argsort(purities)[::-1]

    # Purity bar chart
    fig.add_trace(
        go.Bar(
            x=[model_names[i] for i in sorted_indices],
            y=[purities[i] for i in sorted_indices],
            marker_color='steelblue',
            name='Purity'
        ),
        row=1, col=1
    )

    # Silhouette bar chart
    fig.add_trace(
        go.Bar(
            x=[model_names[i] for i in sorted_indices],
            y=[silhouettes[i] for i in sorted_indices],
            marker_color='coral',
            name='Silhouette'
        ),
        row=1, col=2
    )

    fig.update_layout(
        title='Embedding Model Comparison',
        width=1400,
        height=500,
        showlegend=False
    )

    fig.update_xaxes(tickangle=45)

    # Save
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig.write_html(output_path)
    logger.info(f"Saved model comparison to {output_path}")

    return fig


def create_summary_report(
    results: List[Dict],
    output_path: str
) -> str:
    """
    Create a text summary report of model evaluation.

    Args:
        results: List of dicts with model evaluation results
        output_path: Path to save report

    Returns:
        Report text
    """
    # Sort by purity
    sorted_results = sorted(results, key=lambda x: x.get('category_purity', 0), reverse=True)

    lines = [
        "=" * 70,
        "EMBEDDING MODEL EVALUATION REPORT",
        "=" * 70,
        "",
        f"Total models evaluated: {len(results)}",
        "",
        "RANKING (by Category Purity):",
        "-" * 70,
        f"{'Rank':<6}{'Model':<45}{'Purity':<12}{'Silhouette':<12}",
        "-" * 70
    ]

    for i, r in enumerate(sorted_results, 1):
        model_name = r['model_name'].split('/')[-1][:42]
        purity = r.get('category_purity', 0)
        silhouette = r.get('silhouette_score', 0)
        lines.append(f"{i:<6}{model_name:<45}{purity:<12.2f}{silhouette:<12.4f}")

    lines.extend([
        "-" * 70,
        "",
        "RECOMMENDATIONS:",
        ""
    ])

    if sorted_results:
        best = sorted_results[0]
        lines.append(f"Best model: {best['model_name']}")
        lines.append(f"  - Category Purity: {best.get('category_purity', 0):.2f}%")
        lines.append(f"  - Silhouette Score: {best.get('silhouette_score', 0):.4f}")

    lines.extend([
        "",
        "=" * 70
    ])

    report = "\n".join(lines)

    # Save
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w') as f:
        f.write(report)
    logger.info(f"Saved summary report to {output_path}")

    return report
