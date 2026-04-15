#!/usr/bin/env python3
"""
Fetch works from Adam Mickiewicz University using the OpenAlex API.
Saves results to uam_works.json file.
"""

import json
import logging
import requests
import time
from typing import List, Dict

# Target scope - UAM (Adam Mickiewicz University)
INSTITUTION_ID = "i59411706"
INSTITUTION_NAME = "Adam Mickiewicz University"
INSTITUTION_FILTER = f"authorships.institutions.lineage:{INSTITUTION_ID}"

# API Configuration
BASE_URL = "https://api.openalex.org/works"
PER_PAGE = 200  # Max allowed by API
RATE_LIMIT_DELAY = 0.01  # Seconds between requests

def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

def author_display_name(author: Dict) -> str:
    """Return a sensible display name for an author entry."""
    if not author:
        return "Unknown"

    name = author.get('display_name')
    if name:
        return name

    alternatives = author.get('display_name_alternatives')
    if isinstance(alternatives, list) and alternatives:
        return alternatives[0]
    if isinstance(alternatives, str) and alternatives:
        return alternatives

    return "Unknown"


def fetch_all_works() -> List[Dict]:
    """Fetch all works for the configured institution using cursor paging."""
    logging.info("Fetching works from OpenAlex API (using cursor paging)...")
    logging.info(f"Institution filter: {INSTITUTION_FILTER}")

    all_works = []
    cursor = "*"  # Start with * to get initial cursor
    page_num = 0
    total_results = None

    while cursor:
        page_num += 1

        # Build API URL with cursor
        url = f"{BASE_URL}?filter={INSTITUTION_FILTER}&sort=cited_by_count:desc&per_page={PER_PAGE}&cursor={cursor}"

        logging.info(f"Fetching page {page_num}...")

        try:
            response = requests.get(url, timeout=30)
            response.raise_for_status()

            data = response.json()

            # Get metadata
            meta = data.get('meta', {})

            if page_num == 1:
                total_results = meta.get('count', 0)
                logging.info(f"Total works matching filter: {total_results:,}")

            # Get next cursor for pagination
            next_cursor = meta.get('next_cursor')

            # Get results
            results = data.get('results', [])

            if not results:
                logging.info("No more results, done!")
                break

            all_works.extend(results)
            logging.info(f"Page {page_num}: Retrieved {len(results)} works (total so far: {len(all_works):,})")

            # Check if we have all results
            if total_results and len(all_works) >= total_results:
                logging.info("Retrieved all works!")
                break

            # Update cursor for next iteration
            cursor = next_cursor

            # If cursor is None or empty, we're done
            if not cursor:
                logging.info("No more pages (cursor is None)")
                break

            # Rate limiting
            time.sleep(RATE_LIMIT_DELAY)

        except requests.exceptions.RequestException as e:
            logging.error(f"API request failed on page {page_num}: {e}")

            if page_num == 1:
                raise
            else:
                logging.warning(f"Stopping at page {page_num-1} due to error")
                break

        except Exception as e:
            logging.error(f"Unexpected error on page {page_num}: {e}")
            break

    logging.info(f"Fetching complete: Retrieved {len(all_works):,} works")
    return all_works

def analyze_works(works: List[Dict]):
    """Show analysis of fetched works."""
    if not works:
        return

    print("\n" + "=" * 70)
    print(f"{INSTITUTION_NAME.upper()} WORKS ANALYSIS")
    print("=" * 70)

    print(f"Total works: {len(works):,}\n")

    # Already sorted by cited_by_count from API
    print("Top 25 most cited works:")
    for i, work in enumerate(works[:25], 1):
        title = work.get('title') or work.get('display_name', 'Untitled')
        if len(title) > 80:
            title = title[:77] + "..."

        citations = work.get('cited_by_count', 0)
        year = work.get('publication_year', 'N/A')
        work_type = work.get('type', 'unknown')
        doi = work.get('doi', '')

        authorships = work.get('authorships', [])

        # First three authors for quick context
        primary_authors = []
        for authorship in authorships[:3]:
            name = author_display_name(authorship.get('author', {}))
            primary_authors.append(name)

        # Authors with UAM affiliation
        uam_authors = []
        for authorship in authorships:
            institutions = authorship.get('institutions', [])
            for inst in institutions:
                inst_id = inst.get('id', '').replace('https://openalex.org/', '')
                if inst_id == INSTITUTION_ID:
                    uam_authors.append(author_display_name(authorship.get('author', {})))
                    break

        print(f"{i:2d}. {title}")
        print(f"     Year: {year} | Citations: {citations:,} | Type: {work_type}")
        if primary_authors:
            authors_str = ', '.join(primary_authors)
            if len(authorships) > 3:
                authors_str += f" (+{len(authorships)-3} more)"
            print(f"     Lead authors: {authors_str}")
        if uam_authors:
            display_authors = uam_authors[:3]
            authors_str = ', '.join(display_authors)
            if len(uam_authors) > 3:
                authors_str += f" (+{len(uam_authors)-3} more)"
            print(f"     UAM-affiliated authors: {authors_str}")
        if doi:
            print(f"     DOI: {doi}")
        print()

    # Statistics
    total_citations = sum(w.get('cited_by_count', 0) for w in works)

    # Publication years
    years = [w.get('publication_year') for w in works if w.get('publication_year')]
    if years:
        oldest_year = min(years)
        newest_year = max(years)
    else:
        oldest_year = newest_year = 'N/A'

    # Work types
    work_types = {}
    for w in works:
        wtype = w.get('type', 'unknown')
        work_types[wtype] = work_types.get(wtype, 0) + 1

    # Open access
    oa_works = sum(1 for w in works if w.get('open_access', {}).get('oa_status') in ['gold', 'green', 'hybrid', 'bronze'])

    print("Statistics:")
    print(f"  Total citations: {total_citations:,}")
    print(f"  Avg citations/work: {total_citations/len(works):.1f}")
    print(f"  Publication years: {oldest_year} - {newest_year}")
    print(f"  Open access works: {oa_works:,} ({oa_works/len(works)*100:.1f}%)")

    print("\n  Top work types:")
    for wtype, count in sorted(work_types.items(), key=lambda x: x[1], reverse=True)[:10]:
        print(f"    {wtype}: {count:,} ({count/len(works)*100:.1f}%)")

def save_to_file(works: List[Dict], filename: str = 'uam_works.json'):
    """Save works to JSON file"""
    try:
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(works, f, indent=2, ensure_ascii=False)
        logging.info(f"Saved {len(works):,} works to {filename}")
    except Exception as e:
        logging.error(f"Error saving to file: {e}")

def main():
    setup_logging()

    logging.info("=" * 70)
    logging.info("WORKS FETCHER - Using OpenAlex API")
    logging.info(f"Institution: {INSTITUTION_NAME} (ID: {INSTITUTION_ID})")
    logging.info("=" * 70)

    try:
        # Fetch all works from API
        works = fetch_all_works()

        if not works:
            logging.warning("No works found!")
            return

        # Save to file
        save_to_file(works)

        # Analyze
        analyze_works(works)

        logging.info("=" * 70)
        logging.info("SUCCESS!")
        logging.info(f"Works fetched: {len(works):,}")
        logging.info(f"Saved to: uam_works.json")
        logging.info("=" * 70)

    except Exception as e:
        logging.error(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
