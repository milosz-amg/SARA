#!/usr/bin/env python3
"""Get detailed info about specific researchers."""

import psycopg2
from psycopg2.extras import RealDictCursor
import json

DB_PARAMS = {
    'host': 'localhost',
    'port': 5432,
    'database': 'devdb',
    'user': 'devuser',
    'password': 'devpass'
}

def get_researcher_details(name):
    """Get detailed information about a researcher."""
    conn = psycopg2.connect(**DB_PARAMS, cursor_factory=RealDictCursor)
    cursor = conn.cursor()

    cursor.execute("""
        SELECT display_name, h_index, cited_by_count, works_count, concepts
        FROM authors
        WHERE display_name = %s;
    """, [name])

    result = cursor.fetchone()

    if result:
        print(f"\n{'='*80}")
        print(f"RESEARCHER: {result['display_name']}")
        print(f"{'='*80}")
        print(f"H-index: {result['h_index']}")
        print(f"Cytowania: {result['cited_by_count']}")
        print(f"Publikacje: {result['works_count']}")

        if result['concepts']:
            concepts = result['concepts']
            if isinstance(concepts, str):
                concepts = json.loads(concepts)
            print(f"\nTop koncepty:")
            for i, c in enumerate(concepts[:10], 1):
                print(f"  {i}. {c.get('display_name', 'Unknown')} (score: {c.get('score', 0):.2f})")

    cursor.close()
    conn.close()

# Check results
names = [
    "Patryk Żywica",
    "Bartlomiej Klin",
    "Peter Trudgill",
    "Wojciech Buszkowski"
]

for name in names:
    get_researcher_details(name)
