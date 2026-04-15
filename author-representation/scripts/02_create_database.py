#!/usr/bin/env python3
"""
Create Database Script

Creates SQLite database from raw JSON files and validates data.
"""

import sys
import os
import json
import sqlite3
import argparse
import logging
from datetime import datetime
from typing import List, Dict
from collections import Counter
import glob

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import configs.config as config

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def create_database(db_path: str):
    """
    Create SQLite database with schema.

    Args:
        db_path: Path to database file
    """
    logger.info(f"Creating database at {db_path}")

    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # Execute schema
    cursor.executescript(config.DB_SCHEMA)

    conn.commit()
    conn.close()

    logger.info("Database schema created successfully")


def load_json_files(raw_data_dir: str) -> List[Dict]:
    """
    Load all paper JSON files from raw data directory.

    Args:
        raw_data_dir: Directory containing JSON files

    Returns:
        List of paper dictionaries
    """
    logger.info(f"Loading JSON files from {raw_data_dir}")

    all_papers = []
    json_files = glob.glob(os.path.join(raw_data_dir, "papers_*.json"))

    if not json_files:
        logger.warning(f"No JSON files found in {raw_data_dir}")
        return []

    logger.info(f"Found {len(json_files)} JSON files")

    for json_file in json_files:
        logger.info(f"Loading {os.path.basename(json_file)}...")
        with open(json_file, 'r', encoding='utf-8') as f:
            papers = json.load(f)
            all_papers.extend(papers)
            logger.info(f"  Loaded {len(papers)} papers")

    logger.info(f"Total papers loaded: {len(all_papers)}")
    return all_papers


def insert_papers(db_path: str, papers: List[Dict], batch_size: int = 1000):
    """
    Insert papers into database.

    Args:
        db_path: Path to database file
        papers: List of paper dictionaries
        batch_size: Number of papers to insert at once
    """
    logger.info(f"Inserting {len(papers)} papers into database...")

    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    insert_sql = """
    INSERT OR REPLACE INTO papers (
        id, title, abstract, authors, categories, primary_category,
        published_date, updated_date, doi, arxiv_url, pdf_url,
        comment, journal_ref
    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """

    successful = 0
    failed = 0

    for i in range(0, len(papers), batch_size):
        batch = papers[i:i + batch_size]
        batch_data = []

        for paper in batch:
            try:
                # Prepare data
                data = (
                    paper['id'],
                    paper['title'],
                    paper.get('abstract'),
                    json.dumps(paper.get('authors', [])),
                    json.dumps(paper.get('categories', [])),
                    paper.get('primary_category'),
                    paper.get('published'),
                    paper.get('updated'),
                    paper.get('doi'),
                    paper.get('arxiv_url'),
                    paper.get('pdf_url'),
                    paper.get('comment'),
                    paper.get('journal_ref'),
                )
                batch_data.append(data)
                successful += 1
            except Exception as e:
                logger.warning(f"Failed to prepare paper {paper.get('id')}: {str(e)}")
                failed += 1
                continue

        # Insert batch
        try:
            cursor.executemany(insert_sql, batch_data)
            conn.commit()
            logger.info(f"Inserted batch {i//batch_size + 1}/{(len(papers) + batch_size - 1)//batch_size}")
        except sqlite3.Error as e:
            logger.error(f"Database error: {str(e)}")
            conn.rollback()

    conn.close()

    logger.info(f"Insertion complete: {successful} successful, {failed} failed")


def generate_validation_report(db_path: str) -> Dict:
    """
    Generate validation report from database.

    Args:
        db_path: Path to database file

    Returns:
        Dictionary with statistics
    """
    logger.info("Generating validation report...")

    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    report = {}

    # Total papers
    cursor.execute("SELECT COUNT(*) FROM papers")
    report['total_papers'] = cursor.fetchone()[0]

    # Papers by category
    cursor.execute("""
        SELECT primary_category, COUNT(*)
        FROM papers
        GROUP BY primary_category
        ORDER BY COUNT(*) DESC
    """)
    report['by_category'] = dict(cursor.fetchall())

    # Papers with abstracts
    cursor.execute("SELECT COUNT(*) FROM papers WHERE abstract IS NOT NULL AND abstract != ''")
    report['papers_with_abstracts'] = cursor.fetchone()[0]

    # Papers with DOI
    cursor.execute("SELECT COUNT(*) FROM papers WHERE doi IS NOT NULL")
    report['papers_with_doi'] = cursor.fetchone()[0]

    # Date range
    cursor.execute("""
        SELECT MIN(published_date), MAX(published_date)
        FROM papers
        WHERE published_date IS NOT NULL
    """)
    min_date, max_date = cursor.fetchone()
    report['date_range'] = {
        'earliest': min_date,
        'latest': max_date
    }

    # Average authors per paper
    cursor.execute("SELECT authors FROM papers")
    author_counts = []
    for row in cursor.fetchall():
        try:
            authors = json.loads(row[0])
            author_counts.append(len(authors))
        except:
            continue
    report['avg_authors_per_paper'] = sum(author_counts) / len(author_counts) if author_counts else 0

    # Average categories per paper
    cursor.execute("SELECT categories FROM papers")
    category_counts = []
    for row in cursor.fetchall():
        try:
            categories = json.loads(row[0])
            category_counts.append(len(categories))
        except:
            continue
    report['avg_categories_per_paper'] = sum(category_counts) / len(category_counts) if category_counts else 0

    conn.close()

    return report


def print_report(report: Dict):
    """Print validation report in a readable format."""
    print("\n" + "="*60)
    print("DATABASE VALIDATION REPORT")
    print("="*60)

    print(f"\nTotal Papers: {report['total_papers']:,}")

    print(f"\nPapers with Abstracts: {report['papers_with_abstracts']:,} "
          f"({100*report['papers_with_abstracts']/report['total_papers']:.1f}%)")

    print(f"Papers with DOI: {report['papers_with_doi']:,} "
          f"({100*report['papers_with_doi']/report['total_papers']:.1f}%)")

    print(f"\nAverage Authors per Paper: {report['avg_authors_per_paper']:.1f}")
    print(f"Average Categories per Paper: {report['avg_categories_per_paper']:.1f}")

    print(f"\nDate Range:")
    print(f"  Earliest: {report['date_range']['earliest']}")
    print(f"  Latest: {report['date_range']['latest']}")

    print(f"\nPapers by Category:")
    for category, count in sorted(report['by_category'].items(), key=lambda x: x[1], reverse=True):
        print(f"  {category}: {count:,}")

    print("\n" + "="*60)


def save_report(report: Dict, output_path: str):
    """Save report as JSON."""
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    logger.info(f"Report saved to {output_path}")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='Create SQLite database from raw JSON files',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Create database from default raw data directory
  python 02_create_database.py

  # Create database from custom directory
  python 02_create_database.py --raw-data-dir ./my_data

  # Force overwrite existing database
  python 02_create_database.py --force
        """
    )

    parser.add_argument(
        '--raw-data-dir',
        type=str,
        default=config.RAW_DATA_DIR,
        help=f'Directory containing raw JSON files (default: {config.RAW_DATA_DIR})'
    )

    parser.add_argument(
        '--db-path',
        type=str,
        default=config.DB_PATH,
        help=f'Path to SQLite database (default: {config.DB_PATH})'
    )

    parser.add_argument(
        '--force',
        action='store_true',
        help='Overwrite existing database'
    )

    parser.add_argument(
        '--batch-size',
        type=int,
        default=config.DB_BATCH_SIZE,
        help=f'Batch size for insertions (default: {config.DB_BATCH_SIZE})'
    )

    parser.add_argument(
        '--report-output',
        type=str,
        default=None,
        help='Path to save validation report JSON (default: None, print only)'
    )

    args = parser.parse_args()

    # Check if database exists
    if os.path.exists(args.db_path) and not args.force:
        logger.error(f"Database already exists at {args.db_path}")
        logger.error("Use --force to overwrite")
        sys.exit(1)

    # Remove existing database if force
    if os.path.exists(args.db_path) and args.force:
        logger.info(f"Removing existing database at {args.db_path}")
        os.remove(args.db_path)

    # Ensure data directory exists
    os.makedirs(os.path.dirname(args.db_path), exist_ok=True)

    logger.info("ArXiv Database Creation Script")
    logger.info(f"Raw data directory: {args.raw_data_dir}")
    logger.info(f"Database path: {args.db_path}")

    start_time = datetime.now()

    # Step 1: Create database
    create_database(args.db_path)

    # Step 2: Load JSON files
    papers = load_json_files(args.raw_data_dir)

    if not papers:
        logger.error("No papers found. Please run 01_fetch_arxiv_papers.py first.")
        sys.exit(1)

    # Step 3: Insert papers
    insert_papers(args.db_path, papers, args.batch_size)

    # Step 4: Generate validation report
    report = generate_validation_report(args.db_path)
    print_report(report)

    # Save report if requested
    if args.report_output:
        save_report(report, args.report_output)

    elapsed = datetime.now() - start_time
    logger.info(f"\nTotal execution time: {elapsed}")
    logger.info("Database created successfully!")


if __name__ == '__main__':
    main()
