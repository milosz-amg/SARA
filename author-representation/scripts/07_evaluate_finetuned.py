#!/usr/bin/env python3
"""
Evaluate Fine-tuned Model

Compares the original BGE-base-en-v1.5 with the fine-tuned version using:
  - PCA clustering (purity, silhouette score)
  - Precision/Recall@k (multi-label retrieval)
  - Fuzzy purity (multi-label aware clustering)
"""

import sys
import os
import sqlite3
import json
import argparse
import logging
from datetime import datetime
from typing import List, Dict, Tuple
from collections import Counter, defaultdict

import numpy as np
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
from sentence_transformers import SentenceTransformer

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from configs.config import DB_PATH, DATA_DIR, PROJECT_ROOT, RESULTS_DIR
from src.category_hierarchy import extract_main_category

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

MODELS_DIR = os.path.join(PROJECT_ROOT, 'models')


def load_test_papers(db_path: str) -> List[Dict]:
    """Load all papers from database with multi-label categories."""
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("""
        SELECT id, title, abstract, primary_category, categories
        FROM papers
        WHERE abstract IS NOT NULL AND abstract != ''
        ORDER BY id
    """)
    papers = []
    for r in cursor.fetchall():
        papers.append({
            'id': r[0],
            'title': r[1],
            'abstract': r[2],
            'primary_category': r[3],
            'categories': json.loads(r[4]) if r[4] else [r[3]]
        })
    conn.close()
    return papers


def generate_embeddings(model: SentenceTransformer, papers: List[Dict], batch_size: int = 128) -> np.ndarray:
    """Generate embeddings for papers."""
    texts = [f"{p['title']}. {p['abstract']}" if p['abstract'] else p['title'] for p in papers]
    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
        normalize_embeddings=True,
        convert_to_numpy=True
    )
    return embeddings


def evaluate_clustering(
    embeddings: np.ndarray,
    categories: np.ndarray,
    multi_categories: List[List[str]],
    k_values: List[int]
) -> Dict:
    """Run PCA + K-means and calculate metrics for multiple k values."""
    scaler = StandardScaler()
    embeddings_scaled = scaler.fit_transform(embeddings)
    pca = PCA(n_components=2, random_state=42)
    coords = pca.fit_transform(embeddings_scaled)

    main_categories = np.array([extract_main_category(c) for c in categories])

    results = {}

    for k in k_values:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = kmeans.fit_predict(coords)

        sil_score = silhouette_score(coords, labels)

        # Subcategory purity (hard)
        sub_correct = 0
        for cluster_id in np.unique(labels):
            mask = labels == cluster_id
            counts = Counter(categories[mask])
            sub_correct += counts.most_common(1)[0][1]
        sub_purity = (sub_correct / len(labels)) * 100

        # Main category purity (hard)
        main_correct = 0
        for cluster_id in np.unique(labels):
            mask = labels == cluster_id
            counts = Counter(main_categories[mask])
            main_correct += counts.most_common(1)[0][1]
        main_purity = (main_correct / len(labels)) * 100

        # Fuzzy purity: paper is "correct" if it shares ANY category
        # with the cluster's dominant category set
        fuzzy_correct = 0
        for cluster_id in np.unique(labels):
            mask = labels == cluster_id
            indices = np.where(mask)[0]
            # Find dominant primary category in cluster
            counts = Counter(categories[mask])
            dominant_cat = counts.most_common(1)[0][0]
            dominant_main = extract_main_category(dominant_cat)
            # Collect all subcategories that map to the dominant main group
            dominant_cats = set()
            for idx in indices:
                for cat in multi_categories[idx]:
                    if extract_main_category(cat) == dominant_main:
                        dominant_cats.add(cat)
            # Paper is correct if ANY of its categories is in dominant set
            for idx in indices:
                paper_cats = set(multi_categories[idx])
                if paper_cats & dominant_cats:
                    fuzzy_correct += 1
        fuzzy_purity = (fuzzy_correct / len(labels)) * 100

        results[k] = {
            'subcategory_purity': sub_purity,
            'main_category_purity': main_purity,
            'fuzzy_purity': fuzzy_purity,
            'silhouette_score': sil_score,
            'variance_explained': float(sum(pca.explained_variance_ratio_))
        }

        logger.info(f"  k={k}: sub={sub_purity:.2f}%, main={main_purity:.2f}%, "
                     f"fuzzy={fuzzy_purity:.2f}%, sil={sil_score:.4f}")

    return results


def precision_recall_at_k(
    embeddings: np.ndarray,
    multi_categories: List[List[str]],
    k_values: List[int] = [1, 3, 5, 10, 20],
    sample_size: int = 2000
) -> Dict:
    """
    Calculate Precision@k and Recall@k using multi-label relevance.

    A neighbor is 'relevant' if it shares ANY category with the query paper.
    Uses sampling for efficiency on large datasets.
    """
    n = len(embeddings)

    # Sample queries for efficiency
    if sample_size and n > sample_size:
        query_indices = np.random.choice(n, sample_size, replace=False)
    else:
        query_indices = np.arange(n)

    # Build category index: paper_idx -> set of categories
    cat_sets = [set(cats) for cats in multi_categories]

    max_k = max(k_values)

    metrics = {}
    for k in k_values:
        metrics[f'P@{k}'] = []
        metrics[f'R@{k}'] = []

    # Process in batches to avoid memory issues
    batch_size = 256
    for batch_start in range(0, len(query_indices), batch_size):
        batch_indices = query_indices[batch_start:batch_start + batch_size]
        batch_embeddings = embeddings[batch_indices]

        # Cosine similarity (embeddings are normalized)
        sims = batch_embeddings @ embeddings.T  # (batch, n)

        for i, qi in enumerate(batch_indices):
            # Zero out self-similarity
            sims[i, qi] = -1.0

            # Get top-k neighbors
            top_k_indices = np.argpartition(-sims[i], max_k)[:max_k]
            top_k_indices = top_k_indices[np.argsort(-sims[i, top_k_indices])]

            # Count total relevant papers
            query_cats = cat_sets[qi]
            total_relevant = sum(1 for j in range(n) if j != qi and query_cats & cat_sets[j])

            if total_relevant == 0:
                continue

            for k in k_values:
                relevant_in_topk = sum(1 for j in top_k_indices[:k] if query_cats & cat_sets[j])
                metrics[f'P@{k}'].append(relevant_in_topk / k)
                metrics[f'R@{k}'].append(relevant_in_topk / total_relevant)

    # Average
    results = {}
    for key, values in metrics.items():
        results[key] = float(np.mean(values)) if values else 0.0

    return results


def main():
    parser = argparse.ArgumentParser(description='Evaluate fine-tuned model vs original')
    parser.add_argument('--original-model', type=str, default='BAAI/bge-base-en-v1.5',
                        help='Original model name')
    parser.add_argument('--finetuned-model', type=str, default=None,
                        help='Path to fine-tuned model')
    parser.add_argument('--batch-size', type=int, default=128,
                        help='Embedding batch size')
    parser.add_argument('--pr-sample', type=int, default=2000,
                        help='Sample size for P/R@k calculation (default: 2000)')

    args = parser.parse_args()

    finetuned_path = args.finetuned_model or os.path.join(MODELS_DIR, 'bge-base-arxiv-finetuned', 'final')

    output_dir = os.path.join(RESULTS_DIR, 'finetuned_comparison')
    os.makedirs(output_dir, exist_ok=True)

    logger.info("=" * 60)
    logger.info("Evaluating Original vs Fine-tuned Model")
    logger.info("=" * 60)
    logger.info(f"Original: {args.original_model}")
    logger.info(f"Fine-tuned: {finetuned_path}")

    # Load papers
    papers = load_test_papers(DB_PATH)
    categories = np.array([p['primary_category'] for p in papers])
    multi_categories = [p['categories'] for p in papers]
    logger.info(f"Loaded {len(papers)} papers")

    k_values = [8, 20, 118]
    pr_k_values = [1, 3, 5, 10, 20]

    np.random.seed(42)

    all_results = {}

    for model_label, model_path in [('original', args.original_model), ('finetuned', finetuned_path)]:
        logger.info(f"\n{'='*40}")
        logger.info(f"Evaluating: {model_label} ({model_path})")
        logger.info(f"{'='*40}")

        model = SentenceTransformer(model_path)
        embeddings = generate_embeddings(model, papers, args.batch_size)

        logger.info("Clustering metrics:")
        cluster_results = evaluate_clustering(embeddings, categories, multi_categories, k_values)

        logger.info("\nPrecision/Recall@k:")
        pr_results = precision_recall_at_k(embeddings, multi_categories, pr_k_values, args.pr_sample)
        for key, val in sorted(pr_results.items()):
            logger.info(f"  {key}: {val:.4f}")

        all_results[model_label] = {
            'clustering': {str(k): v for k, v in cluster_results.items()},
            'retrieval': pr_results
        }

        del model
        import torch
        torch.cuda.empty_cache()

    # Save comparison JSON
    comparison = {
        'original_model': args.original_model,
        'finetuned_model': finetuned_path,
        'num_papers': len(papers),
        'evaluated_at': datetime.now().isoformat(),
        'results': all_results
    }

    comparison_path = os.path.join(output_dir, 'comparison.json')
    with open(comparison_path, 'w') as f:
        json.dump(comparison, f, indent=2)

    # Print comparison table
    print("\n" + "=" * 90)
    print("COMPARISON: Original vs Fine-tuned BGE-base")
    print("=" * 90)

    # Clustering metrics
    print(f"\n{'CLUSTERING METRICS'}")
    print(f"{'Metric':<30} {'Original':<15} {'Fine-tuned':<15} {'Diff':<15}")
    print("-" * 75)

    for k in k_values:
        print(f"\nk={k} clusters:")
        orig_c = all_results['original']['clustering'][str(k)]
        ft_c = all_results['finetuned']['clustering'][str(k)]
        for metric in ['main_category_purity', 'subcategory_purity', 'fuzzy_purity', 'silhouette_score']:
            orig_val = orig_c[metric]
            ft_val = ft_c[metric]
            diff = ft_val - orig_val
            sign = "+" if diff > 0 else ""
            if 'purity' in metric:
                print(f"  {metric:<28} {orig_val:<15.2f} {ft_val:<15.2f} {sign}{diff:<15.2f}")
            else:
                print(f"  {metric:<28} {orig_val:<15.4f} {ft_val:<15.4f} {sign}{diff:<15.4f}")

    # Retrieval metrics
    print(f"\n{'RETRIEVAL METRICS (multi-label)'}")
    print(f"{'Metric':<30} {'Original':<15} {'Fine-tuned':<15} {'Diff':<15}")
    print("-" * 75)

    orig_r = all_results['original']['retrieval']
    ft_r = all_results['finetuned']['retrieval']
    for k in pr_k_values:
        for metric_type in ['P', 'R']:
            key = f'{metric_type}@{k}'
            orig_val = orig_r[key]
            ft_val = ft_r[key]
            diff = ft_val - orig_val
            sign = "+" if diff > 0 else ""
            print(f"  {key:<28} {orig_val:<15.4f} {ft_val:<15.4f} {sign}{diff:<15.4f}")

    print("=" * 90)

    # Save markdown report
    md_lines = [
        "# Fine-tuned BGE-base Evaluation",
        "",
        f"**Original model:** {args.original_model}",
        f"**Fine-tuned model:** {finetuned_path}",
        f"**Total papers:** {len(papers)}",
        f"**Evaluated at:** {datetime.now().strftime('%Y-%m-%d %H:%M')}",
        "",
        "## Clustering Metrics",
        "",
        "| k | Metric | Original | Fine-tuned | Diff |",
        "|---|--------|----------|------------|------|"
    ]

    for k in k_values:
        orig_c = all_results['original']['clustering'][str(k)]
        ft_c = all_results['finetuned']['clustering'][str(k)]
        for metric in ['main_category_purity', 'subcategory_purity', 'fuzzy_purity', 'silhouette_score']:
            orig_val = orig_c[metric]
            ft_val = ft_c[metric]
            diff = ft_val - orig_val
            sign = "+" if diff > 0 else ""
            if 'purity' in metric:
                md_lines.append(f"| {k} | {metric} | {orig_val:.2f}% | {ft_val:.2f}% | {sign}{diff:.2f}% |")
            else:
                md_lines.append(f"| {k} | {metric} | {orig_val:.4f} | {ft_val:.4f} | {sign}{diff:.4f} |")

    md_lines.extend([
        "",
        "## Retrieval Metrics (Multi-Label)",
        "",
        "| Metric | Original | Fine-tuned | Diff |",
        "|--------|----------|------------|------|"
    ])

    for k in pr_k_values:
        for metric_type in ['P', 'R']:
            key = f'{metric_type}@{k}'
            orig_val = orig_r[key]
            ft_val = ft_r[key]
            diff = ft_val - orig_val
            sign = "+" if diff > 0 else ""
            md_lines.append(f"| {key} | {orig_val:.4f} | {ft_val:.4f} | {sign}{diff:.4f} |")

    md_lines.extend([
        "",
        "## Notes",
        "",
        "- **Fuzzy purity**: Paper is correct if it shares ANY category with cluster's dominant group",
        "- **P@k/R@k**: A neighbor is relevant if it shares ANY category with the query (multi-label)",
        f"- P/R@k computed on {args.pr_sample} sampled queries",
    ])

    md_path = os.path.join(output_dir, 'comparison_report.md')
    with open(md_path, 'w') as f:
        f.write('\n'.join(md_lines))

    logger.info(f"\nResults saved to: {output_dir}")
    logger.info(f"  - {comparison_path}")
    logger.info(f"  - {md_path}")


if __name__ == '__main__':
    main()
