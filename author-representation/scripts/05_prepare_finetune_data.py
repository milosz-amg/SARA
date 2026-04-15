#!/usr/bin/env python3
"""
Prepare Fine-tuning Dataset

Two modes:
  --mode pairs   : anchor/positive pairs for MultipleNegativesRankingLoss (legacy)
  --mode scored  : sentence1/sentence2/score for CoSENTLoss (hierarchical + fuzzy)

The 'scored' mode uses multi-label categories and hierarchical distance
to generate pairs with continuous similarity labels (0.0-1.0).

Outputs train/val/test splits as HuggingFace Datasets.
"""

import sys
import os
import sqlite3
import json
import argparse
import logging
import random
from collections import defaultdict
from typing import List, Dict, Tuple

import numpy as np
from datasets import Dataset, DatasetDict

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from configs.config import DB_PATH, DATA_DIR
from src.category_hierarchy import (
    extract_main_category, extract_archive, multilabel_similarity
)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

FINETUNE_DIR = os.path.join(DATA_DIR, 'finetune')


def load_papers(db_path: str, min_papers: int = 0, max_papers: int = 0) -> List[Dict]:
    """Load papers from database, filtering categories by size.

    Args:
        db_path: Path to SQLite database
        min_papers: Skip categories with fewer papers (0 = no filter)
        max_papers: Cap papers per category (0 = no cap)
    """
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    cursor.execute("""
        SELECT id, title, abstract, primary_category, categories
        FROM papers
        WHERE abstract IS NOT NULL AND abstract != ''
        ORDER BY id
    """)

    all_papers = []
    for row in cursor.fetchall():
        all_papers.append({
            'id': row[0],
            'title': row[1],
            'abstract': row[2],
            'primary_category': row[3],
            'categories': json.loads(row[4]) if row[4] else [row[3]]
        })
    conn.close()
    logger.info(f"Loaded {len(all_papers)} papers from database")

    # Group by category for filtering
    by_category = defaultdict(list)
    for paper in all_papers:
        by_category[paper['primary_category']].append(paper)

    # Apply min/max filters
    papers = []
    skipped_small = []
    capped = []

    for category, cat_papers in sorted(by_category.items()):
        if min_papers > 0 and len(cat_papers) < min_papers:
            skipped_small.append((category, len(cat_papers)))
            continue

        if max_papers > 0 and len(cat_papers) > max_papers:
            random.shuffle(cat_papers)
            capped.append((category, len(cat_papers), max_papers))
            cat_papers = cat_papers[:max_papers]

        papers.extend(cat_papers)

    if skipped_small:
        logger.info(f"Skipped {len(skipped_small)} categories with <{min_papers} papers:")
        for cat, count in skipped_small:
            logger.info(f"  {cat}: {count}")

    if capped:
        logger.info(f"Capped {len(capped)} categories to {max_papers} papers:")
        for cat, orig, cap in capped:
            logger.info(f"  {cat}: {orig} -> {cap}")

    logger.info(f"After filtering: {len(papers)} papers in {len(by_category) - len(skipped_small)} categories")
    return papers


def combine_text(title: str, abstract: str) -> str:
    """Combine title and abstract for embedding."""
    if abstract:
        return f"{title}. {abstract}"
    return title


def split_papers(
    papers: List[Dict],
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    seed: int = 42
) -> Tuple[List[Dict], List[Dict], List[Dict]]:
    """
    Split papers into train/val/test, stratified by category.

    Args:
        papers: List of paper dicts
        train_ratio: Fraction for training
        val_ratio: Fraction for validation (test = 1 - train - val)
        seed: Random seed

    Returns:
        train_papers, val_papers, test_papers
    """
    random.seed(seed)
    np.random.seed(seed)

    # Group by category
    by_category = defaultdict(list)
    for paper in papers:
        by_category[paper['primary_category']].append(paper)

    train_papers = []
    val_papers = []
    test_papers = []

    for category, cat_papers in by_category.items():
        random.shuffle(cat_papers)

        n = len(cat_papers)
        n_train = max(1, int(n * train_ratio))
        n_val = max(1, int(n * val_ratio))

        train_papers.extend(cat_papers[:n_train])
        val_papers.extend(cat_papers[n_train:n_train + n_val])
        test_papers.extend(cat_papers[n_train + n_val:])

    logger.info(f"Split: train={len(train_papers)}, val={len(val_papers)}, test={len(test_papers)}")

    return train_papers, val_papers, test_papers


def generate_pairs(
    papers: List[Dict],
    pairs_per_paper: int = 5,
    seed: int = 42
) -> List[Dict]:
    """
    Generate anchor/positive pairs from papers in the same subcategory.

    Args:
        papers: List of paper dicts
        pairs_per_paper: Number of positive pairs per anchor paper
        seed: Random seed

    Returns:
        List of dicts with 'anchor' and 'positive' keys
    """
    random.seed(seed)

    # Group by category
    by_category = defaultdict(list)
    for paper in papers:
        by_category[paper['primary_category']].append(paper)

    pairs = []

    for category, cat_papers in by_category.items():
        if len(cat_papers) < 2:
            continue

        for paper in cat_papers:
            # Get other papers in same category (excluding self)
            others = [p for p in cat_papers if p['id'] != paper['id']]

            if not others:
                continue

            # Sample positive papers
            n_positives = min(pairs_per_paper, len(others))
            positives = random.sample(others, n_positives)

            anchor_text = combine_text(paper['title'], paper['abstract'])

            for pos in positives:
                positive_text = combine_text(pos['title'], pos['abstract'])
                pairs.append({
                    'anchor': anchor_text,
                    'positive': positive_text
                })

    random.shuffle(pairs)
    logger.info(f"Generated {len(pairs)} pairs from {len(papers)} papers")

    return pairs


def generate_triplets(
    papers: List[Dict],
    triplets_per_paper: int = 3,
    seed: int = 42
) -> List[Dict]:
    """
    Generate anchor/positive/negative triplets for TripletEvaluator.

    Args:
        papers: List of paper dicts
        triplets_per_paper: Number of triplets per anchor paper
        seed: Random seed

    Returns:
        List of dicts with 'anchor', 'positive', 'negative' keys
    """
    random.seed(seed)

    # Group by category
    by_category = defaultdict(list)
    for paper in papers:
        by_category[paper['primary_category']].append(paper)

    all_papers = list(papers)
    triplets = []

    for category, cat_papers in by_category.items():
        if len(cat_papers) < 2:
            continue

        # Papers from OTHER categories (for negatives)
        other_papers = [p for p in all_papers if p['primary_category'] != category]

        if not other_papers:
            continue

        for paper in cat_papers:
            others_same = [p for p in cat_papers if p['id'] != paper['id']]

            if not others_same:
                continue

            n_triplets = min(triplets_per_paper, len(others_same))
            positives = random.sample(others_same, n_triplets)
            negatives = random.sample(other_papers, n_triplets)

            anchor_text = combine_text(paper['title'], paper['abstract'])

            for pos, neg in zip(positives, negatives):
                triplets.append({
                    'anchor': anchor_text,
                    'positive': combine_text(pos['title'], pos['abstract']),
                    'negative': combine_text(neg['title'], neg['abstract'])
                })

    random.shuffle(triplets)
    logger.info(f"Generated {len(triplets)} triplets from {len(papers)} papers")

    return triplets


def generate_scored_pairs(
    papers: List[Dict],
    pairs_per_paper: int = 8,
    seed: int = 42
) -> List[Dict]:
    """
    Generate pairs with continuous similarity scores for CoSENTLoss.

    For each anchor, samples papers from different distance tiers:
      - 3 from same subcategory (score ~1.0)
      - 2 from same archive, different subcategory (score ~0.5-0.7)
      - 2 from same main group, different archive (score ~0.2-0.4)
      - 1 from different main group (score ~0.0)

    Actual scores are computed via multilabel_similarity() for fuzzy labels.

    Returns:
        List of dicts with 'sentence1', 'sentence2', 'score' keys
    """
    random.seed(seed)

    # Build indices for stratified sampling
    by_subcategory = defaultdict(list)
    by_archive = defaultdict(list)
    by_main_group = defaultdict(list)
    all_papers_list = list(papers)

    for paper in papers:
        pcat = paper['primary_category']
        by_subcategory[pcat].append(paper)
        by_archive[extract_archive(pcat)].append(paper)
        by_main_group[extract_main_category(pcat)].append(paper)

    pairs = []
    tier_counts = {'same_sub': 0, 'same_archive': 0, 'same_group': 0, 'diff_group': 0}

    for paper in papers:
        anchor_text = combine_text(paper['title'], paper['abstract'])
        pcat = paper['primary_category']
        archive = extract_archive(pcat)
        group = extract_main_category(pcat)

        sampled = []

        # Tier 1: same subcategory (3 pairs)
        same_sub = [p for p in by_subcategory[pcat] if p['id'] != paper['id']]
        n = min(3, len(same_sub))
        if n > 0:
            sampled.extend(random.sample(same_sub, n))
            tier_counts['same_sub'] += n

        # Tier 2: same archive, different subcategory (2 pairs)
        same_arch = [p for p in by_archive[archive]
                     if p['id'] != paper['id'] and p['primary_category'] != pcat]
        n = min(2, len(same_arch))
        if n > 0:
            sampled.extend(random.sample(same_arch, n))
            tier_counts['same_archive'] += n

        # Tier 3: same main group, different archive (2 pairs)
        same_grp = [p for p in by_main_group[group]
                    if p['id'] != paper['id'] and extract_archive(p['primary_category']) != archive]
        n = min(2, len(same_grp))
        if n > 0:
            sampled.extend(random.sample(same_grp, n))
            tier_counts['same_group'] += n

        # Tier 4: different main group (1 pair)
        diff_grp = [p for p in all_papers_list
                    if p['id'] != paper['id'] and extract_main_category(p['primary_category']) != group]
        n = min(1, len(diff_grp))
        if n > 0:
            sampled.extend(random.sample(diff_grp, n))
            tier_counts['diff_group'] += n

        # Compute similarity scores and create pairs
        for other in sampled:
            score = multilabel_similarity(paper['categories'], other['categories'])
            pairs.append({
                'sentence1': anchor_text,
                'sentence2': combine_text(other['title'], other['abstract']),
                'score': float(score)
            })

    random.shuffle(pairs)
    logger.info(f"Generated {len(pairs)} scored pairs from {len(papers)} papers")
    logger.info(f"  Tier distribution: {dict(tier_counts)}")

    # Log score distribution
    scores = [p['score'] for p in pairs]
    logger.info(f"  Score stats: min={min(scores):.3f}, max={max(scores):.3f}, "
                f"mean={np.mean(scores):.3f}, median={np.median(scores):.3f}")

    return pairs


def generate_ir_eval_data(papers: List[Dict]) -> Dict:
    """
    Generate Information Retrieval evaluation data.

    For each paper, finds all other papers sharing any category (multi-label).
    Returns queries, corpus, and relevant_docs mappings for
    InformationRetrievalEvaluator.
    """
    queries = {}
    corpus = {}
    relevant_docs = {}

    # Build corpus
    for paper in papers:
        corpus[paper['id']] = combine_text(paper['title'], paper['abstract'])

    # Build category -> paper_id index
    cat_to_papers = defaultdict(set)
    for paper in papers:
        for cat in paper['categories']:
            cat_to_papers[cat].add(paper['id'])

    # For each paper, find relevant docs (sharing any category)
    for paper in papers:
        queries[paper['id']] = combine_text(paper['title'], paper['abstract'])
        relevant = set()
        for cat in paper['categories']:
            relevant |= cat_to_papers[cat]
        relevant.discard(paper['id'])
        if relevant:
            relevant_docs[paper['id']] = relevant

    logger.info(f"IR eval data: {len(queries)} queries, {len(corpus)} corpus docs, "
                f"avg {np.mean([len(v) for v in relevant_docs.values()]):.1f} relevant per query")

    return {'queries': queries, 'corpus': corpus, 'relevant_docs': relevant_docs}


def main():
    parser = argparse.ArgumentParser(description='Prepare fine-tuning dataset')
    parser.add_argument('--mode', type=str, default='scored', choices=['pairs', 'scored'],
                        help='pairs=MNR loss (legacy), scored=CoSENTLoss with hierarchy (default: scored)')
    parser.add_argument('--pairs-per-paper', type=int, default=8,
                        help='Number of pairs per paper (default: 8 for scored, 5 for pairs)')
    parser.add_argument('--triplets-per-paper', type=int, default=3,
                        help='Number of triplets per paper for evaluation (default: 3)')
    parser.add_argument('--train-ratio', type=float, default=0.7,
                        help='Training split ratio (default: 0.7)')
    parser.add_argument('--val-ratio', type=float, default=0.15,
                        help='Validation split ratio (default: 0.15)')
    parser.add_argument('--min-papers', type=int, default=10,
                        help='Skip categories with fewer papers (default: 10)')
    parser.add_argument('--max-papers', type=int, default=300,
                        help='Cap papers per category to prevent domination (default: 300)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed (default: 42)')
    parser.add_argument('--output-dir', type=str, default=None,
                        help='Output directory (default: data/finetune/)')

    args = parser.parse_args()

    output_dir = args.output_dir or FINETUNE_DIR
    os.makedirs(output_dir, exist_ok=True)

    logger.info("=" * 60)
    logger.info("Preparing fine-tuning dataset")
    logger.info(f"Mode: {args.mode}")
    logger.info(f"Database: {DB_PATH}")
    logger.info(f"Output: {output_dir}")
    logger.info(f"Min papers per category: {args.min_papers}")
    logger.info(f"Max papers per category: {args.max_papers}")
    logger.info("=" * 60)

    # Load papers (with category size filtering)
    random.seed(args.seed)
    papers = load_papers(DB_PATH, min_papers=args.min_papers, max_papers=args.max_papers)

    # Print category distribution
    cat_counts = defaultdict(int)
    for p in papers:
        cat_counts[p['primary_category']] += 1

    logger.info(f"\nTotal categories: {len(cat_counts)}")
    logger.info(f"Papers per category: min={min(cat_counts.values())}, "
                f"max={max(cat_counts.values())}, "
                f"avg={sum(cat_counts.values())/len(cat_counts):.1f}")

    # Split papers
    train_papers, val_papers, test_papers = split_papers(
        papers, args.train_ratio, args.val_ratio, args.seed
    )

    if args.mode == 'scored':
        # --- CoSENTLoss mode: scored pairs with hierarchical similarity ---
        logger.info("\nGenerating scored training pairs...")
        train_pairs = generate_scored_pairs(train_papers, args.pairs_per_paper, args.seed)

        logger.info("\nGenerating scored validation pairs...")
        val_pairs = generate_scored_pairs(val_papers, args.pairs_per_paper, args.seed + 1)

        # Training dataset (scored pairs for CoSENTLoss)
        train_dataset = Dataset.from_dict({
            'sentence1': [p['sentence1'] for p in train_pairs],
            'sentence2': [p['sentence2'] for p in train_pairs],
            'score': [p['score'] for p in train_pairs]
        })
        val_dataset = Dataset.from_dict({
            'sentence1': [p['sentence1'] for p in val_pairs],
            'sentence2': [p['sentence2'] for p in val_pairs],
            'score': [p['score'] for p in val_pairs]
        })

        n_train_pairs = len(train_pairs)
        n_val_pairs = len(val_pairs)

    else:
        # --- Legacy MNR mode: anchor/positive pairs ---
        logger.info("\nGenerating training pairs...")
        train_pairs = generate_pairs(train_papers, args.pairs_per_paper, args.seed)

        logger.info("\nGenerating validation pairs...")
        val_pairs = generate_pairs(val_papers, args.pairs_per_paper, args.seed + 1)

        train_dataset = Dataset.from_dict({
            'anchor': [p['anchor'] for p in train_pairs],
            'positive': [p['positive'] for p in train_pairs]
        })
        val_dataset = Dataset.from_dict({
            'anchor': [p['anchor'] for p in val_pairs],
            'positive': [p['positive'] for p in val_pairs]
        })

        n_train_pairs = len(train_pairs)
        n_val_pairs = len(val_pairs)

    # Generate evaluation triplets (for TripletEvaluator - used in both modes)
    logger.info("\nGenerating evaluation triplets...")
    val_triplets = generate_triplets(val_papers, args.triplets_per_paper, args.seed + 2)
    test_triplets = generate_triplets(test_papers, args.triplets_per_paper, args.seed + 3)

    val_triplet_dataset = Dataset.from_dict({
        'anchor': [t['anchor'] for t in val_triplets],
        'positive': [t['positive'] for t in val_triplets],
        'negative': [t['negative'] for t in val_triplets]
    })
    test_triplet_dataset = Dataset.from_dict({
        'anchor': [t['anchor'] for t in test_triplets],
        'positive': [t['positive'] for t in test_triplets],
        'negative': [t['negative'] for t in test_triplets]
    })

    # Generate IR evaluation data (multi-label relevant docs)
    logger.info("\nGenerating IR evaluation data...")
    test_ir_data = generate_ir_eval_data(test_papers)

    # Save datasets
    logger.info("\nSaving datasets...")
    train_dataset.save_to_disk(os.path.join(output_dir, 'train'))
    val_dataset.save_to_disk(os.path.join(output_dir, 'val'))
    val_triplet_dataset.save_to_disk(os.path.join(output_dir, 'val_triplets'))
    test_triplet_dataset.save_to_disk(os.path.join(output_dir, 'test_triplets'))

    # Save IR eval data (convert sets to lists for JSON)
    ir_serializable = {
        'queries': test_ir_data['queries'],
        'corpus': test_ir_data['corpus'],
        'relevant_docs': {k: list(v) for k, v in test_ir_data['relevant_docs'].items()}
    }
    with open(os.path.join(output_dir, 'ir_eval_data.json'), 'w') as f:
        json.dump(ir_serializable, f)
    logger.info(f"Saved IR eval data ({len(test_ir_data['queries'])} queries)")

    # Save metadata
    metadata = {
        'mode': args.mode,
        'total_papers': len(papers),
        'train_papers': len(train_papers),
        'val_papers': len(val_papers),
        'test_papers': len(test_papers),
        'train_pairs': n_train_pairs,
        'val_pairs': n_val_pairs,
        'val_triplets': len(val_triplets),
        'test_triplets': len(test_triplets),
        'pairs_per_paper': args.pairs_per_paper,
        'triplets_per_paper': args.triplets_per_paper,
        'train_ratio': args.train_ratio,
        'val_ratio': args.val_ratio,
        'seed': args.seed,
        'min_papers_filter': args.min_papers,
        'max_papers_cap': args.max_papers,
        'num_categories': len(cat_counts)
    }

    with open(os.path.join(output_dir, 'metadata.json'), 'w') as f:
        json.dump(metadata, f, indent=2)

    # Print summary
    print("\n" + "=" * 60)
    print(f"DATASET SUMMARY (mode={args.mode})")
    print("=" * 60)
    print(f"Total papers:       {len(papers)}")
    print(f"Categories:         {len(cat_counts)}")
    print(f"")
    print(f"Train papers:       {len(train_papers)}")
    print(f"Val papers:         {len(val_papers)}")
    print(f"Test papers:        {len(test_papers)}")
    print(f"")
    print(f"Train pairs:        {n_train_pairs}")
    print(f"Val pairs:          {n_val_pairs}")
    print(f"Val triplets:       {len(val_triplets)}")
    print(f"Test triplets:      {len(test_triplets)}")
    print(f"IR test queries:    {len(test_ir_data['queries'])}")
    print(f"")
    print(f"Saved to: {output_dir}")
    print("=" * 60)


if __name__ == '__main__':
    main()
