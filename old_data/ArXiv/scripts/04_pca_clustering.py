#!/usr/bin/env python3
"""
PCA Clustering Script

Performs PCA dimensionality reduction and K-means clustering on embeddings.
Generates visualizations and cluster statistics.
"""

import sys
import os
import sqlite3
import json
import argparse
import logging
from datetime import datetime
from typing import List, Dict, Tuple, Optional
import numpy as np
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
from collections import Counter

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from old_data.ArXiv.config import (
    DB_PATH, EMBEDDINGS_DIR, RESULTS_DIR, EMBEDDING_MODELS,
    MIN_CLUSTERS, MAX_CLUSTERS
)

# PCA components
PCA_COMPONENTS_2D = 2
PCA_COMPONENTS_3D = 3
from old_data.ArXiv.scripts.utils.visualization import (
    plot_pca_2d, plot_pca_3d, plot_elbow_curve, plot_category_distribution
)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_embeddings(model_name: str) -> Tuple[np.ndarray, Dict]:
    """Load embeddings and metadata for a model."""
    safe_name = model_name.replace('/', '-').replace('\\', '-')
    embed_dir = os.path.join(EMBEDDINGS_DIR, safe_name)

    embeddings_path = os.path.join(embed_dir, 'embeddings.npy')
    metadata_path = os.path.join(embed_dir, 'metadata.json')

    if not os.path.exists(embeddings_path):
        raise FileNotFoundError(f"Embeddings not found: {embeddings_path}")

    embeddings = np.load(embeddings_path)

    metadata = {}
    if os.path.exists(metadata_path):
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)

    logger.info(f"Loaded embeddings: {embeddings.shape}")
    return embeddings, metadata


def load_paper_metadata(db_path: str) -> Dict:
    """Load paper titles, IDs, and categories from database."""
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    cursor.execute("""
        SELECT id, title, primary_category
        FROM papers
        WHERE abstract IS NOT NULL AND abstract != ''
        ORDER BY id
    """)

    papers = cursor.fetchall()
    conn.close()

    return {
        'ids': [p[0] for p in papers],
        'titles': [p[1] for p in papers],
        'categories': [p[2] for p in papers]
    }


def perform_pca(
    embeddings: np.ndarray,
    n_components: int,
    standardize: bool = True
) -> Tuple[np.ndarray, PCA, Dict]:
    """
    Perform PCA on embeddings.

    Args:
        embeddings: Input embeddings (N, D)
        n_components: Number of PCA components
        standardize: Whether to standardize before PCA

    Returns:
        PCA coordinates, fitted PCA object, variance info
    """
    logger.info(f"Performing PCA with {n_components} components...")

    if standardize:
        scaler = StandardScaler()
        embeddings_scaled = scaler.fit_transform(embeddings)
    else:
        embeddings_scaled = embeddings

    pca = PCA(n_components=n_components, random_state=42)
    coords = pca.fit_transform(embeddings_scaled)

    variance_info = {
        'explained_variance_ratio': pca.explained_variance_ratio_.tolist(),
        'total_variance_explained': sum(pca.explained_variance_ratio_),
        'variance_pc1': pca.explained_variance_ratio_[0],
        'variance_pc2': pca.explained_variance_ratio_[1] if n_components > 1 else 0,
        'variance_pc3': pca.explained_variance_ratio_[2] if n_components > 2 else 0
    }

    logger.info(f"Total variance explained: {variance_info['total_variance_explained']:.2%}")

    return coords, pca, variance_info


def find_optimal_k(
    embeddings: np.ndarray,
    k_range: Tuple[int, int],
    sample_size: Optional[int] = None
) -> Tuple[int, List[float], List[float]]:
    """
    Find optimal number of clusters using elbow method and silhouette score.

    Args:
        embeddings: Input embeddings
        k_range: (min_k, max_k) range to test
        sample_size: If set, sample this many points for faster computation

    Returns:
        Optimal k, list of inertias, list of silhouette scores
    """
    logger.info(f"Finding optimal k in range {k_range}...")

    # Sample if needed for large datasets
    if sample_size and len(embeddings) > sample_size:
        indices = np.random.choice(len(embeddings), sample_size, replace=False)
        embeddings_sample = embeddings[indices]
    else:
        embeddings_sample = embeddings

    k_values = list(range(k_range[0], k_range[1] + 1))
    inertias = []
    silhouettes = []

    for k in k_values:
        logger.info(f"  Testing k={k}...")
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = kmeans.fit_predict(embeddings_sample)

        inertias.append(kmeans.inertia_)
        silhouettes.append(silhouette_score(embeddings_sample, labels))

    # Find optimal k using silhouette score (higher is better)
    optimal_idx = np.argmax(silhouettes)
    optimal_k = k_values[optimal_idx]

    logger.info(f"Optimal k: {optimal_k} (silhouette: {silhouettes[optimal_idx]:.4f})")

    return optimal_k, inertias, silhouettes


def perform_clustering(
    embeddings: np.ndarray,
    n_clusters: int
) -> Tuple[np.ndarray, KMeans, float]:
    """
    Perform K-means clustering.

    Args:
        embeddings: Input embeddings
        n_clusters: Number of clusters

    Returns:
        Cluster labels, fitted KMeans object, silhouette score
    """
    logger.info(f"Performing K-means clustering with k={n_clusters}...")

    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    labels = kmeans.fit_predict(embeddings)

    sil_score = silhouette_score(embeddings, labels)
    logger.info(f"Silhouette score: {sil_score:.4f}")

    return labels, kmeans, sil_score


def extract_main_category(subcategory: str) -> str:
    """
    Extract main ArXiv group (8 official groups per ArXiv Category Taxonomy).

    Groups:
        cs       - Computer Science
        econ     - Economics
        eess     - Electrical Engineering and Systems Science
        math     - Mathematics
        physics  - Physics (includes astro-ph, cond-mat, gr-qc, hep-*, math-ph,
                   nlin, nucl-*, physics, quant-ph)
        q-bio    - Quantitative Biology
        q-fin    - Quantitative Finance
        stat     - Statistics
    """
    # All archives that belong to the Physics group
    PHYSICS_ARCHIVES = [
        'astro-ph', 'cond-mat', 'gr-qc', 'hep-ex', 'hep-lat', 'hep-ph',
        'hep-th', 'math-ph', 'nlin', 'nucl-ex', 'nucl-th', 'physics',
        'quant-ph'
    ]

    # Get the archive prefix (before the dot)
    prefix = subcategory.split('.')[0] if '.' in subcategory else subcategory

    # Map to 8 official groups
    if prefix in PHYSICS_ARCHIVES:
        return 'physics'
    elif prefix in ['cs', 'econ', 'eess', 'math', 'q-bio', 'q-fin', 'stat']:
        return prefix
    else:
        return prefix  # Fallback


def calculate_category_purity(
    cluster_labels: np.ndarray,
    categories: np.ndarray
) -> Tuple[float, Dict]:
    """
    Calculate category purity for clusters.

    Args:
        cluster_labels: Cluster assignment for each paper
        categories: Ground truth category for each paper

    Returns:
        Overall purity percentage, per-cluster details
    """
    unique_clusters = np.unique(cluster_labels)
    cluster_details = {}
    total_correct = 0
    total_papers = len(cluster_labels)

    for cluster_id in unique_clusters:
        mask = cluster_labels == cluster_id
        cluster_categories = categories[mask]
        cluster_size = len(cluster_categories)

        # Count categories in this cluster
        category_counts = Counter(cluster_categories)
        dominant_category = category_counts.most_common(1)[0]
        dominant_count = dominant_category[1]

        purity = dominant_count / cluster_size
        total_correct += dominant_count

        cluster_details[int(cluster_id)] = {
            'size': cluster_size,
            'dominant_category': dominant_category[0],
            'dominant_count': dominant_count,
            'purity': purity,
            'category_distribution': dict(category_counts)
        }

    overall_purity = (total_correct / total_papers) * 100

    return overall_purity, cluster_details


def process_model(
    model_name: str,
    db_path: str,
    output_dir: str,
    k_clusters: Optional[int] = None
) -> Dict:
    """
    Process a single embedding model: PCA + clustering + visualization.

    Args:
        model_name: Name of the embedding model
        db_path: Path to database
        output_dir: Directory for outputs
        k_clusters: Fixed number of clusters (if None, find optimal)

    Returns:
        Results dictionary
    """
    logger.info(f"\n{'='*60}")
    logger.info(f"Processing model: {model_name}")
    logger.info(f"{'='*60}")

    safe_name = model_name.replace('/', '-').replace('\\', '-')
    model_output_dir = os.path.join(output_dir, safe_name)
    os.makedirs(model_output_dir, exist_ok=True)

    # Load data
    embeddings, embed_metadata = load_embeddings(model_name)
    paper_metadata = load_paper_metadata(db_path)

    # Verify alignment
    if len(embeddings) != len(paper_metadata['ids']):
        logger.warning(f"Mismatch: {len(embeddings)} embeddings vs {len(paper_metadata['ids'])} papers")
        # Use minimum
        min_len = min(len(embeddings), len(paper_metadata['ids']))
        embeddings = embeddings[:min_len]
        for key in paper_metadata:
            paper_metadata[key] = paper_metadata[key][:min_len]

    categories = np.array(paper_metadata['categories'])

    # PCA 2D
    coords_2d, pca_2d, variance_2d = perform_pca(embeddings, PCA_COMPONENTS_2D)

    # PCA 3D
    coords_3d, pca_3d, variance_3d = perform_pca(embeddings, PCA_COMPONENTS_3D)

    # Find optimal k or use provided
    if k_clusters is None:
        optimal_k, inertias, silhouettes = find_optimal_k(
            coords_2d, (MIN_CLUSTERS, MAX_CLUSTERS)
        )

        # Save elbow curve
        k_values = list(range(MIN_CLUSTERS, MAX_CLUSTERS + 1))
        plot_elbow_curve(
            k_values, inertias, silhouettes, optimal_k,
            f"Elbow Analysis - {model_name}",
            os.path.join(model_output_dir, "elbow_curve.html")
        )
    else:
        optimal_k = k_clusters
        inertias, silhouettes = [], []

    # Perform clustering on PCA coordinates
    cluster_labels, kmeans, sil_score = perform_clustering(coords_2d, optimal_k)

    # Calculate category purity on subcategories (118 categories)
    category_purity_sub, cluster_details = calculate_category_purity(cluster_labels, categories)

    # Calculate category purity on main categories (e.g., cs.AI -> cs)
    main_categories = np.array([extract_main_category(cat) for cat in categories])
    category_purity_main, cluster_details_main = calculate_category_purity(cluster_labels, main_categories)

    logger.info(f"Subcategory purity (118 categories): {category_purity_sub:.2f}%")
    logger.info(f"Main category purity (~10 categories): {category_purity_main:.2f}%")

    # Create visualizations
    viz_metadata = {
        **paper_metadata,
        **variance_2d
    }

    # 2D plot colored by category
    plot_pca_2d(
        coords_2d, categories, viz_metadata,
        f"PCA 2D - {model_name} (by Category)",
        os.path.join(model_output_dir, "pca_2d_category.html"),
        color_by='category'
    )

    # 2D plot colored by cluster
    plot_pca_2d(
        coords_2d, cluster_labels.astype(str), viz_metadata,
        f"PCA 2D - {model_name} (by Cluster)",
        os.path.join(model_output_dir, "pca_2d_cluster.html"),
        color_by='cluster'
    )

    # 3D plot
    viz_metadata_3d = {
        **paper_metadata,
        **variance_3d
    }
    plot_pca_3d(
        coords_3d, categories, viz_metadata_3d,
        f"PCA 3D - {model_name}",
        os.path.join(model_output_dir, "pca_3d.html"),
        color_by='category'
    )

    # Category distribution heatmap
    plot_category_distribution(
        cluster_labels, categories,
        f"Category Distribution - {model_name}",
        os.path.join(model_output_dir, "category_distribution.html")
    )

    # Save results (convert numpy types to native Python for JSON serialization)
    results = {
        'model_name': model_name,
        'num_papers': int(len(embeddings)),
        'embedding_dimension': int(embeddings.shape[1]),
        'num_clusters': int(optimal_k),
        'category_purity_subcategory': float(category_purity_sub),
        'category_purity_main': float(category_purity_main),
        'category_purity': float(category_purity_sub),  # Keep for backward compatibility
        'silhouette_score': float(sil_score),
        'variance_explained_2d': float(variance_2d['total_variance_explained']),
        'variance_explained_3d': float(variance_3d['total_variance_explained']),
        'cluster_details_subcategory': {k: {kk: (float(vv) if isinstance(vv, (np.floating, float)) else int(vv) if isinstance(vv, (np.integer, int)) else vv) for kk, vv in v.items()} for k, v in cluster_details.items()},
        'cluster_details_main': {k: {kk: (float(vv) if isinstance(vv, (np.floating, float)) else int(vv) if isinstance(vv, (np.integer, int)) else vv) for kk, vv in v.items()} for k, v in cluster_details_main.items()},
        'processed_at': datetime.now().isoformat()
    }

    results_path = os.path.join(model_output_dir, "clustering_results.json")
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2, default=lambda x: float(x) if isinstance(x, np.floating) else int(x) if isinstance(x, np.integer) else x)
    logger.info(f"Saved results to {results_path}")

    # Save cluster assignments
    assignments_path = os.path.join(model_output_dir, "cluster_assignments.npy")
    np.save(assignments_path, cluster_labels)

    # Save PCA coordinates
    np.save(os.path.join(model_output_dir, "pca_2d_coords.npy"), coords_2d)
    np.save(os.path.join(model_output_dir, "pca_3d_coords.npy"), coords_3d)

    logger.info(f"\n✓ Completed processing {model_name}")
    logger.info(f"  Subcategory Purity (118 cats): {category_purity_sub:.2f}%")
    logger.info(f"  Main Category Purity (~10 cats): {category_purity_main:.2f}%")
    logger.info(f"  Silhouette Score: {sil_score:.4f}")
    logger.info(f"  Clusters: {optimal_k}")

    return results


def process_all_models(
    db_path: str,
    output_dir: str,
    k_clusters: Optional[int] = None
) -> List[Dict]:
    """Process all configured embedding models."""
    all_results = []

    # Get available models (those with embeddings)
    available_models = []
    for model_name in EMBEDDING_MODELS.keys():
        safe_name = model_name.replace('/', '-').replace('\\', '-')
        embed_path = os.path.join(EMBEDDINGS_DIR, safe_name, 'embeddings.npy')
        if os.path.exists(embed_path):
            available_models.append(model_name)
        else:
            logger.warning(f"Skipping {model_name} - embeddings not found")

    logger.info(f"Processing {len(available_models)} models with embeddings")

    for model_name in available_models:
        try:
            results = process_model(model_name, db_path, output_dir, k_clusters)
            all_results.append(results)
        except Exception as e:
            logger.error(f"Failed to process {model_name}: {e}")
            import traceback
            traceback.print_exc()

    # Save combined results
    combined_path = os.path.join(output_dir, "all_models_results.json")
    with open(combined_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    logger.info(f"\nSaved combined results to {combined_path}")

    # Print summary
    summary_lines = []
    summary_lines.append("=" * 90)
    summary_lines.append("SUMMARY - Model Comparison")
    summary_lines.append("=" * 90)
    summary_lines.append(f"{'Model':<40} {'Sub-Cat%':<12} {'Main-Cat%':<12} {'Silhouette':<12}")
    summary_lines.append("-" * 90)

    sorted_results = sorted(all_results, key=lambda x: x['category_purity_main'], reverse=True)
    for r in sorted_results:
        name = r['model_name'].split('/')[-1][:37]
        summary_lines.append(f"{name:<40} {r['category_purity_subcategory']:<12.2f} {r['category_purity_main']:<12.2f} {r['silhouette_score']:<12.4f}")

    summary_lines.append("=" * 90)
    summary_lines.append(f"\nBest model (by main category): {sorted_results[0]['model_name']}")
    summary_lines.append(f"  Main Category Purity: {sorted_results[0]['category_purity_main']:.2f}%")
    summary_lines.append(f"  Subcategory Purity: {sorted_results[0]['category_purity_subcategory']:.2f}%")

    # Print to console
    for line in summary_lines:
        print(line)

    # Save as text file
    summary_text_path = os.path.join(output_dir, "model_comparison_summary.txt")
    with open(summary_text_path, 'w') as f:
        f.write('\n'.join(summary_lines))
    logger.info(f"\nSaved summary to {summary_text_path}")

    # Save as markdown table
    md_lines = [
        "# ArXiv Embedding Models - Evaluation Results",
        "",
        f"**Total models evaluated:** {len(all_results)}",
        f"**Total papers:** {sorted_results[0]['num_papers']}",
        "",
        "## Ranking by Main Category Purity",
        "",
        "| Rank | Model | Subcategory Purity | Main Category Purity | Silhouette Score | Clusters |",
        "|------|-------|-------------------|---------------------|-----------------|----------|"
    ]

    for i, r in enumerate(sorted_results, 1):
        name = r['model_name']
        md_lines.append(
            f"| {i} | {name} | {r['category_purity_subcategory']:.2f}% | "
            f"{r['category_purity_main']:.2f}% | {r['silhouette_score']:.4f} | {r['num_clusters']} |"
        )

    md_lines.extend([
        "",
        "## Best Model",
        "",
        f"**{sorted_results[0]['model_name']}**",
        "",
        f"- Main Category Purity: **{sorted_results[0]['category_purity_main']:.2f}%**",
        f"- Subcategory Purity: {sorted_results[0]['category_purity_subcategory']:.2f}%",
        f"- Silhouette Score: {sorted_results[0]['silhouette_score']:.4f}",
        f"- Embedding Dimension: {sorted_results[0]['embedding_dimension']}",
        f"- Optimal Clusters: {sorted_results[0]['num_clusters']}",
        "",
        "## Notes",
        "",
        "- **Subcategory Purity**: Measured on 118 ArXiv subcategories (cs.AI, cs.LG, math.ST, etc.)",
        "- **Main Category Purity**: Measured on ~10 main categories (cs, math, physics, etc.)",
        "- **Silhouette Score**: Cluster quality metric (-1 to 1, higher is better)",
        "",
        "Lower subcategory purity is expected due to the large number of fine-grained categories."
    ])

    summary_md_path = os.path.join(output_dir, "model_comparison_summary.md")
    with open(summary_md_path, 'w') as f:
        f.write('\n'.join(md_lines))
    logger.info(f"Saved markdown summary to {summary_md_path}")

    return all_results


def main():
    parser = argparse.ArgumentParser(description='PCA Clustering for ArXiv Embeddings')
    parser.add_argument('--model', type=str, help='Specific model to process')
    parser.add_argument('--all-models', action='store_true', help='Process all available models')
    parser.add_argument('--k-clusters', type=int, help='Fixed number of clusters (default: auto)')
    parser.add_argument('--output-dir', type=str, default=None, help='Output directory')

    args = parser.parse_args()

    # Setup output directory
    output_dir = args.output_dir or os.path.join(RESULTS_DIR, 'pca_visualizations')
    os.makedirs(output_dir, exist_ok=True)

    logger.info("ArXiv PCA Clustering Script")
    logger.info(f"Database: {DB_PATH}")
    logger.info(f"Output directory: {output_dir}")

    if args.all_models:
        results = process_all_models(DB_PATH, output_dir, args.k_clusters)
    elif args.model:
        results = process_model(args.model, DB_PATH, output_dir, args.k_clusters)
    else:
        print("Usage:")
        print("  python 04_pca_clustering.py --model BAAI/bge-large-en-v1.5")
        print("  python 04_pca_clustering.py --all-models")
        print("  python 04_pca_clustering.py --all-models --k-clusters 10")
        sys.exit(1)


if __name__ == '__main__':
    main()
