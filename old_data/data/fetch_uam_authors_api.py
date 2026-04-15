#!/usr/bin/env python3
"""
Fetch authors from Adam Mickiewicz University using the OpenAlex API.
Saves results to uam_authors.json file.
"""

import json
import logging
import requests
import time
from typing import List, Dict

# Target scope - UAM (Adam Mickiewicz University)
INSTITUTION_ID = "i59411706"
INSTITUTION_NAME = "Adam Mickiewicz University"
INSTITUTION_FILTER = f"last_known_institutions.id:{INSTITUTION_ID}"

# API Configuration
BASE_URL = "https://api.openalex.org/authors"
PER_PAGE = 200  # Max allowed by API
RATE_LIMIT_DELAY = 0.1  # Seconds between requests (be nice to the API)

def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

def fetch_all_authors() -> List[Dict]:
    """Fetch all authors for the configured institution using cursor paging."""
    logging.info("Fetching authors from OpenAlex API (using cursor paging)...")
    logging.info(f"Institution filter: {INSTITUTION_FILTER}")

    all_authors = []
    cursor = "*"  # Start with * to get initial cursor
    page_num = 0
    total_results = None

    while cursor:
        page_num += 1

        # Build API URL with cursor
        url = f"{BASE_URL}?filter={INSTITUTION_FILTER}&sort=works_count:desc&per_page={PER_PAGE}&cursor={cursor}"

        logging.info(f"Fetching page {page_num}...")

        try:
            response = requests.get(url, timeout=30)
            response.raise_for_status()

            data = response.json()

            # Get metadata
            meta = data.get('meta', {})

            if page_num == 1:
                total_results = meta.get('count', 0)
                logging.info(f"Total authors matching filter: {total_results:,}")

            # Get next cursor for pagination
            next_cursor = meta.get('next_cursor')

            # Get results
            results = data.get('results', [])

            if not results:
                logging.info("No more results, done!")
                break

            all_authors.extend(results)
            logging.info(f"Page {page_num}: Retrieved {len(results)} authors (total so far: {len(all_authors):,})")

            # Check if we have all results
            if total_results and len(all_authors) >= total_results:
                logging.info("Retrieved all authors!")
                break

            # Update cursor for next iteration
            cursor = next_cursor

            # If cursor is None or empty, we're done
            if not cursor:
                logging.info("No more pages (cursor is None)")
                break

            # Rate limiting - be nice to the API
            time.sleep(RATE_LIMIT_DELAY)

        except requests.exceptions.RequestException as e:
            logging.error(f"API request failed on page {page_num}: {e}")

            # Retry logic for failed requests
            if page_num == 1:
                # If first page fails, give up
                raise
            else:
                # Otherwise, stop and return what we have
                logging.warning(f"Stopping at page {page_num-1} due to error")
                break

        except Exception as e:
            logging.error(f"Unexpected error on page {page_num}: {e}")
            break

    logging.info(f"Fetching complete: Retrieved {len(all_authors):,} authors")
    return all_authors

def analyze_authors(authors: List[Dict]):
    """Show analysis of fetched authors."""
    if not authors:
        return

    print("\n" + "=" * 70)
    print(f"{INSTITUTION_NAME.upper()} AUTHORS ANALYSIS")
    print("=" * 70)

    print(f"Total authors: {len(authors):,}\n")

    # Already sorted by works_count from API
    print("Top 25 authors (by works count):")
    for i, author in enumerate(authors[:25], 1):
        name = author.get('display_name', 'Unknown')
        citations = author.get('cited_by_count', 0)
        works = author.get('works_count', 0)
        h_index = author.get('summary_stats', {}).get('h_index', 0)
        orcid = author.get('orcid', '')

        print(f"{i:2d}. {name}")
        print(f"     {citations:,} citations | {works:,} works | h-index: {h_index}")
        if orcid:
            print(f"     ORCID: {orcid}")
        print()

    # Statistics
    total_citations = sum(a.get('cited_by_count', 0) for a in authors)
    total_works = sum(a.get('works_count', 0) for a in authors)
    authors_with_orcid = sum(1 for a in authors if a.get('orcid'))

    print(f"Statistics:")
    print(f"  Total citations: {total_citations:,}")
    print(f"  Total works: {total_works:,}")
    print(f"  Avg citations/author: {total_citations/len(authors):.1f}")
    print(f"  Avg works/author: {total_works/len(authors):.1f}")
    print(f"  Authors with ORCID: {authors_with_orcid:,} ({authors_with_orcid/len(authors)*100:.1f}%)")

def save_to_file(authors: List[Dict], filename: str = 'uam_authors.json'):
    """Save authors to JSON file"""
    try:
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(authors, f, indent=2, ensure_ascii=False)
        logging.info(f"Saved {len(authors):,} authors to {filename}")
    except Exception as e:
        logging.error(f"Error saving to file: {e}")

def main():
    setup_logging()

    logging.info("=" * 70)
    logging.info("AUTHORS FETCHER - Using OpenAlex API")
    logging.info(f"Institution: {INSTITUTION_NAME} (ID: {INSTITUTION_ID})")
    logging.info("=" * 70)

    try:
        # Fetch all authors from API
        authors = fetch_all_authors()

        if not authors:
            logging.warning("No authors found!")
            return

        # Save to file
        save_to_file(authors)

        # Analyze
        analyze_authors(authors)

        logging.info("=" * 70)
        logging.info("SUCCESS!")
        logging.info(f"Authors fetched: {len(authors):,}")
        logging.info(f"Saved to: uam_authors.json")
        logging.info("=" * 70)

    except Exception as e:
        logging.error(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
