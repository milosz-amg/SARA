#!/usr/bin/env python3
"""
Fetch ArXiv Papers Script

Fetches papers from ArXiv API by category and saves to JSON files.
Follows patterns from data/fetch_uam_authors_api.py
"""

import sys
import os
import json
import argparse
import logging
from datetime import datetime
from typing import List, Dict
from tqdm import tqdm

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.arxiv_client import ArXivAPIClient, ArXivAPIError
import configs.config as config

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f'fetch_arxiv_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


def save_papers_to_json(papers: List[Dict], category: str, output_dir: str):
    """
    Save papers to JSON file.

    Args:
        papers: List of paper dictionaries
        category: ArXiv category
        output_dir: Output directory path
    """
    os.makedirs(output_dir, exist_ok=True)

    filename = f"papers_{category.replace('/', '_')}.json"
    filepath = os.path.join(output_dir, filename)

    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(papers, f, indent=2, ensure_ascii=False)

    logger.info(f"Saved {len(papers)} papers to {filepath}")


def fetch_category_papers(
    category: str,
    limit: int,
    output_dir: str,
    batch_size: int = 1000
) -> List[Dict]:
    """
    Fetch papers for a specific category.

    Args:
        category: ArXiv category (e.g., 'cs', 'physics')
        limit: Maximum number of papers to fetch
        output_dir: Directory to save JSON files
        batch_size: Papers per API request

    Returns:
        List of paper dictionaries
    """
    logger.info(f"Starting fetch for category: {category} (limit: {limit})")

    client = ArXivAPIClient(
        rate_limit_delay=config.RATE_LIMIT_DELAY,
        max_retries=config.MAX_RETRIES,
        backoff_factor=config.BACKOFF_FACTOR
    )

    all_papers = []
    start = 0
    start_time = datetime.now()

    # Create progress bar
    pbar = tqdm(total=limit, desc=f"Fetching {category}", unit="papers")

    try:
        while len(all_papers) < limit:
            try:
                # Fetch batch
                result = client.search_papers(
                    category=category,
                    start=start,
                    max_results=min(batch_size, limit - len(all_papers)),
                    sort_by="submittedDate",
                    sort_order="descending"
                )

                papers = result['papers']
                if not papers:
                    logger.warning(f"No more papers available for {category}")
                    break

                all_papers.extend(papers)
                pbar.update(len(papers))

                # Calculate ETA
                elapsed = (datetime.now() - start_time).total_seconds()
                rate = len(all_papers) / elapsed if elapsed > 0 else 0
                remaining = limit - len(all_papers)
                eta_seconds = remaining / rate if rate > 0 else 0

                pbar.set_postfix({
                    'rate': f'{rate:.1f} papers/s',
                    'ETA': f'{eta_seconds/60:.1f} min'
                })

                # Check if we've reached the limit
                if len(all_papers) >= limit:
                    all_papers = all_papers[:limit]
                    break

                # Check if there are more results
                if len(papers) < batch_size:
                    logger.info(f"Reached end of available papers for {category}")
                    break

                start += batch_size

            except ArXivAPIError as e:
                logger.error(f"API error: {str(e)}")
                logger.info("Retrying after 10 seconds...")
                import time
                time.sleep(10)
                continue

    finally:
        pbar.close()

    # Save papers
    if all_papers:
        save_papers_to_json(all_papers, category, output_dir)
        logger.info(f"Successfully fetched {len(all_papers)} papers for {category}")
    else:
        logger.warning(f"No papers fetched for {category}")

    return all_papers


def fetch_all_categories(categories: Dict[str, int], output_dir: str):
    """
    Fetch papers for all configured categories.

    Args:
        categories: Dictionary of category -> limit
        output_dir: Directory to save JSON files
    """
    logger.info(f"Fetching papers for {len(categories)} categories")
    logger.info(f"Total target: {sum(categories.values())} papers")

    total_fetched = 0

    for category, limit in categories.items():
        logger.info(f"\n{'='*60}")
        logger.info(f"Processing category: {category}")
        logger.info(f"{'='*60}\n")

        papers = fetch_category_papers(
            category=category,
            limit=limit,
            output_dir=output_dir
        )

        total_fetched += len(papers)

        logger.info(f"Completed {category}: {len(papers)} papers")
        logger.info(f"Total progress: {total_fetched}/{sum(categories.values())}\n")

    logger.info(f"\n{'='*60}")
    logger.info(f"All categories completed!")
    logger.info(f"Total papers fetched: {total_fetched}")
    logger.info(f"{'='*60}")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='Fetch papers from ArXiv API',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Fetch 10000 papers from cs category
  python 01_fetch_arxiv_papers.py --category cs --limit 10000

  # Fetch papers for all configured categories
  python 01_fetch_arxiv_papers.py --all-categories

  # Fetch with custom output directory
  python 01_fetch_arxiv_papers.py --category physics --limit 5000 --output-dir ./my_data
        """
    )

    parser.add_argument(
        '--category',
        type=str,
        help='ArXiv category to fetch (e.g., cs, physics, math)'
    )

    parser.add_argument(
        '--limit',
        type=int,
        help='Maximum number of papers to fetch for the category'
    )

    parser.add_argument(
        '--all-categories',
        action='store_true',
        help='Fetch papers for all configured categories'
    )

    parser.add_argument(
        '--output-dir',
        type=str,
        default=config.RAW_DATA_DIR,
        help=f'Output directory for JSON files (default: {config.RAW_DATA_DIR})'
    )

    parser.add_argument(
        '--batch-size',
        type=int,
        default=1000,
        help='Papers per API request (default: 1000, max: 2000)'
    )

    args = parser.parse_args()

    # Validate arguments
    if not args.all_categories and not args.category:
        parser.error("Either --category or --all-categories must be specified")

    if args.category and not args.limit:
        parser.error("--limit is required when using --category")

    # Ensure output directory exists
    os.makedirs(args.output_dir, exist_ok=True)

    # Start fetching
    logger.info("ArXiv Paper Fetching Script")
    logger.info(f"Output directory: {args.output_dir}")
    logger.info(f"Batch size: {args.batch_size}")
    logger.info(f"Rate limit: {config.RATE_LIMIT_DELAY}s between requests\n")

    start_time = datetime.now()

    if args.all_categories:
        fetch_all_categories(config.ARXIV_CATEGORIES, args.output_dir)
    else:
        fetch_category_papers(
            category=args.category,
            limit=args.limit,
            output_dir=args.output_dir,
            batch_size=args.batch_size
        )

    elapsed = datetime.now() - start_time
    logger.info(f"\nTotal execution time: {elapsed}")

    # Calculate expected time warning
    if args.all_categories:
        total_papers = sum(config.ARXIV_CATEGORIES.values())
        estimated_requests = total_papers / args.batch_size
        estimated_time_hours = (estimated_requests * config.RATE_LIMIT_DELAY) / 3600
        logger.info(f"\nNote: Due to ArXiv's 3-second rate limit,")
        logger.info(f"fetching {total_papers} papers may take ~{estimated_time_hours:.1f} hours")


if __name__ == '__main__':
    main()
