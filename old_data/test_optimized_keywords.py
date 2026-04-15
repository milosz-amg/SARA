#!/usr/bin/env python3
"""
Quick test of optimized keyword extraction (using metadata instead of narratives).
"""

import sys
sys.path.insert(0, '/home/jakub/Projekty/SARA/OpenAlex/scripts')

import json
import logging
import time
from llm_generator import DB_PARAMS, load_llm_model, generate_text
import psycopg2
from psycopg2.extras import RealDictCursor

logging.basicConfig(level=logging.INFO, format='%(message)s')

def get_db_connection():
    return psycopg2.connect(**DB_PARAMS, cursor_factory=RealDictCursor)

def fetch_sample_authors(n=3):
    """Fetch sample authors with concepts."""
    conn = get_db_connection()
    cursor = conn.cursor()

    cursor.execute("""
        SELECT id, display_name, works_count, h_index, cited_by_count,
               last_known_institution_name, concepts
        FROM authors
        WHERE concepts IS NOT NULL
        ORDER BY cited_by_count DESC
        LIMIT %s
    """, (n,))

    authors = cursor.fetchall()
    cursor.close()
    conn.close()

    return [dict(row) for row in authors]

def build_keyword_prompt(author):
    """Build prompt from metadata."""
    concepts = author.get('concepts', [])
    if isinstance(concepts, str):
        concepts = json.loads(concepts)

    top_concepts = [c.get('display_name', '') for c in concepts[:10] if c.get('display_name')]
    concepts_str = ', '.join(top_concepts)

    prompt = f"""Generate 5-10 specific research keywords for this researcher as a JSON array.

Researcher: {author.get('display_name', 'Unknown')}
Field: {author.get('last_known_institution_name', 'Unknown')}
Research areas: {concepts_str}
Publications: {author.get('works_count', 0)}, h-index: {author.get('h_index', 0)}

Keywords JSON:"""

    return prompt

def extract_keywords(author) -> list:
    """Extract keywords from metadata."""
    prompt = build_keyword_prompt(author)

    response = generate_text(prompt, max_new_tokens=80, temperature=0.3)

    # Parse keywords
    try:
        start_idx = response.find('[')
        end_idx = response.rfind(']') + 1

        if start_idx != -1 and end_idx > start_idx:
            json_str = response[start_idx:end_idx]
            keywords = json.loads(json_str)

            if isinstance(keywords, list):
                keywords = [kw.strip().lower() for kw in keywords if isinstance(kw, str) and kw.strip()]
                return keywords[:15]
    except Exception as e:
        logging.error(f"Parse error: {e}")

    return []

def main():
    print("="*80)
    print("OPTIMIZED KEYWORD EXTRACTION TEST (using metadata)")
    print("="*80)

    # Load model
    print("\nLoading LLM model...")
    start = time.time()
    load_llm_model()
    print(f"✅ Model loaded in {time.time()-start:.1f}s\n")

    # Fetch samples
    print("Fetching sample authors...")
    authors = fetch_sample_authors(3)
    print(f"✅ Found {len(authors)} authors\n")

    # Test keyword extraction
    total_time = 0
    for i, author in enumerate(authors, 1):
        print("="*80)
        print(f"[{i}/{len(authors)}] {author['display_name']}")
        print("-"*80)

        # Show metadata
        concepts = json.loads(author['concepts']) if isinstance(author['concepts'], str) else author['concepts']
        top_concepts = [c.get('display_name') for c in concepts[:5]]

        print(f"Institution: {author['last_known_institution_name']}")
        print(f"Metrics: {author['works_count']} works, h-index {author['h_index']}")
        print(f"Top concepts: {', '.join(top_concepts)}")
        print()

        start = time.time()
        keywords = extract_keywords(author)
        elapsed = time.time() - start
        total_time += elapsed

        print(f"⏱️  Generation time: {elapsed:.2f}s")
        print(f"Keywords ({len(keywords)}):")
        for kw in keywords:
            print(f"  • {kw}")
        print()

    print("="*80)
    print("✅ TEST COMPLETED")
    print(f"Average time per author: {total_time/len(authors):.2f}s")
    print(f"Estimated rate: {len(authors)/total_time:.2f} authors/sec")
    print("="*80)

if __name__ == "__main__":
    main()
