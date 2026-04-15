#!/usr/bin/env python3
"""
Quick searches to test specific researcher queries.
"""

from old_data.semantic_search import search_similar_authors
import psycopg2
from psycopg2.extras import RealDictCursor

DB_PARAMS = {
    'host': 'localhost',
    'port': 5432,
    'database': 'devdb',
    'user': 'devuser',
    'password': 'devpass'
}

def search_by_name(name):
    """Search for a specific researcher by name."""
    print("\n" + "="*100)
    print(f"🔍 KIM JEST: {name}?")
    print("="*100)

    conn = psycopg2.connect(**DB_PARAMS, cursor_factory=RealDictCursor)
    cursor = conn.cursor()

    # SQL search by name
    cursor.execute("""
        SELECT
            display_name,
            h_index,
            cited_by_count,
            works_count,
            last_known_institution_id,
            concepts
        FROM authors
        WHERE display_name ILIKE %s
        LIMIT 10;
    """, [f'%{name}%'])

    results = cursor.fetchall()

    if results:
        print(f"\n✅ Znaleziono {len(results)} wyników:\n")
        for i, r in enumerate(results, 1):
            print(f"{i}. {r['display_name']}")
            print(f"   h-index: {r['h_index']}, cytowania: {r['cited_by_count']}, publikacje: {r['works_count']}")

            # Show institution ID
            if r['last_known_institution_id']:
                print(f"   Institution ID: {r['last_known_institution_id']}")

            # Parse top concepts
            if r['concepts']:
                try:
                    import json
                    concepts = json.loads(r['concepts'])
                    if concepts and len(concepts) > 0:
                        top_concepts = [c.get('display_name', '') for c in concepts[:5]]
                        print(f"   Koncepty: {', '.join(top_concepts)}")
                except:
                    pass
            print()
    else:
        print("❌ Nie znaleziono takiego naukowca w bazie")

    cursor.close()
    conn.close()


def semantic_search_test(query, description, institution=None):
    """Test semantic search."""
    print("\n" + "="*100)
    print(f"🔍 {description}")
    print(f"Query: '{query}'")
    if institution:
        print(f"Institution filter: {institution}")
    print("="*100)

    try:
        results = search_similar_authors(
            query=query,
            top_k=10,
            boost_metrics=True,
            institution=institution,
            min_citations=0,
            min_works=0
        )

        if results:
            print(f"\n✅ Znaleziono {len(results)} wyników:\n")
            for i, r in enumerate(results[:5], 1):
                print(f"{i}. {r['display_name']}")
                print(f"   h={r['h_index']}, cytowania={r['cited_by_count']}, publikacje={r['works_count']}")
                print(f"   similarity={r['similarity']:.3f}, hybrid_score={r.get('hybrid_score', 0):.3f}")

                # Show institution ID if available
                if r.get('last_known_institution_id'):
                    print(f"   Institution ID: {r['last_known_institution_id']}")
                print()
        else:
            print("❌ Brak wyników")

    except Exception as e:
        print(f"❌ Błąd: {e}")


def main():
    print("="*100)
    print(" QUICK SEARCH TEST")
    print("="*100)

    # 1. Kim jest Patryk Żywica?
    search_by_name("Patryk Żywica")

    # Also try variations
    search_by_name("Zywica")

    # 2. Naukowiec zajmujący się empatią robotów
    semantic_search_test(
        query="robot empathy affective computing human-robot interaction emotional intelligence",
        description="Naukowiec zajmujący się EMPATIĄ ROBOTÓW"
    )

    # 3. Naukowiec z UAM zajmujący się ewaluacją systemów dialogowych
    semantic_search_test(
        query="dialogue system evaluation conversational AI chatbot assessment NLP",
        description="Naukowiec z UAM - EWALUACJA SYSTEMÓW DIALOGOWYCH",
        institution="Adam Mickiewicz"
    )

    print("\n" + "="*100)
    print(" SEARCH COMPLETE")
    print("="*100)


if __name__ == "__main__":
    main()
