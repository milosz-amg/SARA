#!/usr/bin/env python3
"""
Generate Embeddings Script

Generates embeddings for ArXiv papers using specified models.
Follows patterns from OpenAlex/scripts/generate_embeddings_fast.py
"""

import sys
import os
import sqlite3
import json
import argparse
import logging
from datetime import datetime
from typing import List, Dict, Tuple, Optional
import time

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.embedding_utils import (
    load_model,
    save_embeddings,
    embeddings_exist,
    combine_text_for_embedding,
    calculate_embedding_stats
)
import configs.config as config

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f'embedding_generation_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


def load_papers_from_db(db_path: str) -> List[Dict]:
    """
    Load papers from SQLite database.

    Args:
        db_path: Path to database file

    Returns:
        List of paper dictionaries with id, title, abstract
    """
    logger.info(f"Loading papers from {db_path}")

    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    cursor.execute("""
        SELECT id, title, abstract, primary_category
        FROM papers
        ORDER BY id
    """)

    papers = []
    for row in cursor.fetchall():
        papers.append({
            'id': row[0],
            'title': row[1],
            'abstract': row[2],
            'primary_category': row[3]
        })

    conn.close()

    logger.info(f"Loaded {len(papers)} papers from database")
    return papers


def update_embedding_models_table(
    db_path: str,
    model_name: str,
    dimension: int,
    num_papers: int,
    avg_time: float
):
    """
    Update embedding_models table in database.

    Args:
        db_path: Path to database file
        model_name: Model name
        dimension: Embedding dimension
        num_papers: Number of papers embedded
        avg_time: Average time per paper in seconds
    """
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    cursor.execute("""
        INSERT OR REPLACE INTO embedding_models
        (model_name, dimension, generated_at, num_papers_embedded, avg_time_per_paper)
        VALUES (?, ?, ?, ?, ?)
    """, (model_name, dimension, datetime.now().isoformat(), num_papers, avg_time))

    conn.commit()
    conn.close()

    logger.info(f"Updated embedding_models table for {model_name}")


def generate_embeddings_for_model(
    model_name: str,
    papers: List[Dict],
    output_dir: str,
    db_path: str,
    batch_size: Optional[int] = None,
    force: bool = False
) -> bool:
    """
    Generate embeddings for a specific model.

    Args:
        model_name: HuggingFace model name
        papers: List of paper dictionaries
        output_dir: Directory to save embeddings
        db_path: Path to database
        batch_size: Batch size (None = auto from config)
        force: Force regeneration even if embeddings exist

    Returns:
        True if successful, False otherwise
    """
    logger.info(f"\n{'='*60}")
    logger.info(f"Processing model: {model_name}")
    logger.info(f"{'='*60}\n")

    # Check if embeddings already exist
    if not force and embeddings_exist(model_name, output_dir):
        logger.info(f"Embeddings already exist for {model_name}, skipping...")
        logger.info("Use --force to regenerate")
        return True

    # Get batch size
    if batch_size is None:
        batch_size = config.get_model_batch_size(model_name)

    # Load model
    start_time = time.time()
    generator = load_model(model_name, batch_size=batch_size)
    embedding_dim = generator.get_embedding_dim()

    logger.info(f"Model dimension: {embedding_dim}")
    logger.info(f"Batch size: {batch_size}")

    # Prepare texts
    logger.info("Preparing texts for embedding...")
    texts = []
    paper_ids = []

    for paper in papers:
        text = combine_text_for_embedding(paper['title'], paper['abstract'])
        texts.append(text)
        paper_ids.append(paper['id'])

    logger.info(f"Prepared {len(texts)} texts")

    # Generate embeddings
    try:
        embeddings = generator.generate_embeddings(texts, show_progress=True)
        generation_time = time.time() - start_time

        # Calculate statistics
        stats = calculate_embedding_stats(embeddings)
        logger.info(f"\nEmbedding statistics:")
        for key, value in stats.items():
            logger.info(f"  {key}: {value}")

        # Save embeddings
        metadata = {
            'model_name': model_name,
            'dimension': embedding_dim,
            'num_papers': len(papers),
            'batch_size': batch_size,
            'generation_time_seconds': generation_time,
            'avg_time_per_paper': generation_time / len(papers),
            'stats': stats
        }

        save_embeddings(embeddings, model_name, output_dir, metadata)

        # Update database
        update_embedding_models_table(
            db_path,
            model_name,
            embedding_dim,
            len(papers),
            generation_time / len(papers)
        )

        logger.info(f"\n✓ Successfully generated embeddings for {model_name}")
        logger.info(f"  Time: {generation_time:.2f}s")
        logger.info(f"  Rate: {len(papers)/generation_time:.2f} papers/s")

        # Clear model from memory
        generator.clear_model()

        return True

    except Exception as e:
        logger.error(f"Failed to generate embeddings for {model_name}: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


def generate_all_models(
    papers: List[Dict],
    output_dir: str,
    db_path: str,
    models: Dict[str, int],
    force: bool = False
):
    """
    Generate embeddings for all configured models.

    Args:
        papers: List of paper dictionaries
        output_dir: Directory to save embeddings
        db_path: Path to database
        models: Dictionary of model_name -> dimension
        force: Force regeneration
    """
    logger.info(f"Generating embeddings for {len(models)} models")
    logger.info(f"Total papers: {len(papers)}")

    successful = []
    failed = []

    for model_name in models.keys():
        success = generate_embeddings_for_model(
            model_name=model_name,
            papers=papers,
            output_dir=output_dir,
            db_path=db_path,
            force=force
        )

        if success:
            successful.append(model_name)
        else:
            failed.append(model_name)

    # Summary
    logger.info(f"\n{'='*60}")
    logger.info(f"SUMMARY")
    logger.info(f"{'='*60}")
    logger.info(f"Successful: {len(successful)}/{len(models)}")
    if successful:
        logger.info(f"  {', '.join(successful)}")
    if failed:
        logger.warning(f"Failed: {len(failed)}/{len(models)}")
        logger.warning(f"  {', '.join(failed)}")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='Generate embeddings for ArXiv papers',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Generate embeddings for a specific model
  python 03_generate_embeddings.py --model BAAI/bge-large-en-v1.5

  # Generate embeddings for all configured models
  python 03_generate_embeddings.py --all-models

  # Force regeneration even if embeddings exist
  python 03_generate_embeddings.py --model BAAI/bge-large-en-v1.5 --force

  # Use custom batch size
  python 03_generate_embeddings.py --model BAAI/bge-large-en-v1.5 --batch-size 64
        """
    )

    parser.add_argument(
        '--model',
        type=str,
        help='Model name to generate embeddings for'
    )

    parser.add_argument(
        '--all-models',
        action='store_true',
        help='Generate embeddings for all configured models'
    )

    parser.add_argument(
        '--db-path',
        type=str,
        default=config.DB_PATH,
        help=f'Path to SQLite database (default: {config.DB_PATH})'
    )

    parser.add_argument(
        '--output-dir',
        type=str,
        default=config.EMBEDDINGS_DIR,
        help=f'Output directory for embeddings (default: {config.EMBEDDINGS_DIR})'
    )

    parser.add_argument(
        '--batch-size',
        type=int,
        default=None,
        help='Batch size (default: auto from config)'
    )

    parser.add_argument(
        '--force',
        action='store_true',
        help='Force regeneration even if embeddings exist'
    )

    args = parser.parse_args()

    # Validate arguments
    if not args.all_models and not args.model:
        parser.error("Either --model or --all-models must be specified")

    if args.model and args.model not in config.EMBEDDING_MODELS:
        logger.warning(f"Model {args.model} not in configured models")
        logger.warning(f"Configured models: {list(config.EMBEDDING_MODELS.keys())}")
        logger.info("Proceeding anyway...")

    # Check if database exists
    if not os.path.exists(args.db_path):
        logger.error(f"Database not found: {args.db_path}")
        logger.error("Please run 02_create_database.py first")
        sys.exit(1)

    # Ensure output directory exists
    os.makedirs(args.output_dir, exist_ok=True)

    logger.info("ArXiv Embedding Generation Script")
    logger.info(f"Database: {args.db_path}")
    logger.info(f"Output directory: {args.output_dir}")

    # Load papers
    papers = load_papers_from_db(args.db_path)

    if not papers:
        logger.error("No papers found in database")
        sys.exit(1)

    start_time = datetime.now()

    # Generate embeddings
    if args.all_models:
        generate_all_models(
            papers=papers,
            output_dir=args.output_dir,
            db_path=args.db_path,
            models=config.EMBEDDING_MODELS,
            force=args.force
        )
    else:
        success = generate_embeddings_for_model(
            model_name=args.model,
            papers=papers,
            output_dir=args.output_dir,
            db_path=args.db_path,
            batch_size=args.batch_size,
            force=args.force
        )
        if not success:
            sys.exit(1)

    elapsed = datetime.now() - start_time
    logger.info(f"\nTotal execution time: {elapsed}")


if __name__ == '__main__':
    main()
