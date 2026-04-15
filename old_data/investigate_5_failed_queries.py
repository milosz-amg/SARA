#!/usr/bin/env python3
"""
Investigate 5 failed queries: Q005, Q007, Q012, Q014, Q018
For each query:
1. Check SQL: Does data exist in database?
2. Check semantic search: Can embeddings find it?
3. Diagnose the problem
"""

import psycopg2
from psycopg2.extras import RealDictCursor
import json
from old_data.semantic_search import get_query_embedding

DB_PARAMS = {
    'host': 'localhost',
    'port': 5432,
    'database': 'devdb',
    'user': 'devuser',
    'password': 'devpass'
}

def investigate_query(conn, query_id, query_text, filters, concepts_to_check):
    """
    Investigate a single failed query.

    Args:
        query_id: e.g. "Q005"
        query_text: The semantic search query
        filters: Dict with min_citations, min_h_index, min_works, year_from
        concepts_to_check: List of concept keywords to search in DB
    """
    print("\n" + "="*100)
    print(f"{query_id}: {query_text}")
    print("="*100)

    cursor = conn.cursor()

    # Step 1: SQL - Check if data exists in database
    print("\n[1] SQL CHECK: Does data exist in database?")
    print("-" * 80)

    # Build concept search conditions
    concept_conditions = " OR ".join([f"concepts::text ILIKE '%{c}%'" for c in concepts_to_check])

    # Determine if this is authors or works search
    is_works_search = 'year_from' in filters
    table = 'works' if is_works_search else 'authors'

    # Build filter conditions
    filter_conditions = []
    if not is_works_search:
        if filters.get('min_citations', 0) > 0:
            filter_conditions.append(f"cited_by_count >= {filters['min_citations']}")
        if filters.get('min_h_index', 0) > 0:
            filter_conditions.append(f"h_index >= {filters['min_h_index']}")
        if filters.get('min_works', 0) > 0:
            filter_conditions.append(f"works_count >= {filters['min_works']}")
    else:
        if filters.get('min_citations', 0) > 0:
            filter_conditions.append(f"cited_by_count >= {filters['min_citations']}")
        if filters.get('year_from'):
            filter_conditions.append(f"publication_year >= {filters['year_from']}")

    # Query 1: Total with concepts (no filters)
    sql_total = f"""
        SELECT COUNT(*) as count
        FROM {table}
        WHERE {concept_conditions};
    """
    cursor.execute(sql_total)
    total_count = cursor.fetchone()['count']
    print(f"Total {table} matching concepts: {total_count}")

    # Query 2: With filters applied
    if filter_conditions:
        filters_str = " AND ".join(filter_conditions)
        sql_filtered = f"""
            SELECT COUNT(*) as count
            FROM {table}
            WHERE ({concept_conditions})
              AND {filters_str};
        """
        cursor.execute(sql_filtered)
        filtered_count = cursor.fetchone()['count']
        print(f"After applying filters: {filtered_count}")
    else:
        filtered_count = total_count

    # Query 3: Get top examples
    if is_works_search:
        sql_examples = f"""
            SELECT title, publication_year, cited_by_count
            FROM {table}
            WHERE {concept_conditions}
            ORDER BY cited_by_count DESC
            LIMIT 5;
        """
        cursor.execute(sql_examples)
        examples = cursor.fetchall()
        print(f"\nTop 5 {table} by citations:")
        for ex in examples:
            title = ex['title'][:70] + "..." if len(ex['title']) > 70 else ex['title']
            print(f"  - {title}")
            print(f"    {ex['publication_year']}, {ex['cited_by_count']} citations")
    else:
        sql_examples = f"""
            SELECT display_name, h_index, cited_by_count, works_count
            FROM {table}
            WHERE {concept_conditions}
            ORDER BY cited_by_count DESC
            LIMIT 5;
        """
        cursor.execute(sql_examples)
        examples = cursor.fetchall()
        print(f"\nTop 5 {table} by citations:")
        for ex in examples:
            print(f"  - {ex['display_name']}: h={ex['h_index']}, cit={ex['cited_by_count']}, works={ex['works_count']}")

    # Step 2: SEMANTIC SEARCH - Can embeddings find it?
    if not is_works_search:  # Only test semantic search for authors
        print("\n[2] SEMANTIC SEARCH CHECK: Can embeddings find it?")
        print("-" * 80)

        try:
            embedding = get_query_embedding(query_text)
            embedding_str = str(embedding)

            # Get top 10 semantic results WITHOUT filters
            sql_semantic = f"""
                SELECT
                    display_name,
                    h_index,
                    cited_by_count,
                    works_count,
                    1 - (embedding <=> %s::vector) as similarity
                FROM authors
                WHERE embedding IS NOT NULL
                ORDER BY embedding <=> %s::vector
                LIMIT 10;
            """
            cursor.execute(sql_semantic, [embedding_str, embedding_str])
            semantic_results = cursor.fetchall()

            print(f"Top 10 semantic matches (NO filters):")
            for i, r in enumerate(semantic_results, 1):
                print(f"  {i}. {r['display_name']} (sim={r['similarity']:.3f}, h={r['h_index']}, cit={r['cited_by_count']})")

            # Check how many of top 10 match expected concepts
            matching_semantic = []
            for r in semantic_results:
                # Check if this author has expected concepts
                check_sql = f"""
                    SELECT COUNT(*) as has_concept
                    FROM authors
                    WHERE display_name = %s
                      AND ({concept_conditions});
                """
                cursor.execute(check_sql, [r['display_name']])
                if cursor.fetchone()['has_concept'] > 0:
                    matching_semantic.append(r)

            print(f"\n→ {len(matching_semantic)}/10 semantic results have expected concepts")
            if matching_semantic:
                print(f"→ Best matching result: {matching_semantic[0]['display_name']} (sim={matching_semantic[0]['similarity']:.3f})")
            else:
                print(f"→ NONE of top 10 semantic results have expected concepts!")

            # Get top 50 semantic results and check how far down we need to go
            sql_semantic_50 = f"""
                SELECT
                    display_name,
                    h_index,
                    cited_by_count,
                    1 - (embedding <=> %s::vector) as similarity
                FROM authors
                WHERE embedding IS NOT NULL
                ORDER BY embedding <=> %s::vector
                LIMIT 50;
            """
            cursor.execute(sql_semantic_50, [embedding_str, embedding_str])
            semantic_50 = cursor.fetchall()

            matching_in_50 = 0
            high_quality_in_50 = 0
            for r in semantic_50:
                check_sql = f"""
                    SELECT COUNT(*) as has_concept
                    FROM authors
                    WHERE display_name = %s
                      AND ({concept_conditions});
                """
                cursor.execute(check_sql, [r['display_name']])
                if cursor.fetchone()['has_concept'] > 0:
                    matching_in_50 += 1
                    # Check if meets filter criteria
                    meets_criteria = True
                    if filters.get('min_citations', 0) > 0 and r['cited_by_count'] < filters['min_citations']:
                        meets_criteria = False
                    if filters.get('min_h_index', 0) > 0 and r['h_index'] < filters['min_h_index']:
                        meets_criteria = False
                    if meets_criteria:
                        high_quality_in_50 += 1

            print(f"→ {matching_in_50}/50 semantic results have expected concepts")
            print(f"→ {high_quality_in_50}/50 meet both concept AND filter criteria")

        except Exception as e:
            print(f"❌ Semantic search error: {e}")
            matching_in_50 = 0
            high_quality_in_50 = 0
    else:
        # For works search, these variables don't apply
        matching_in_50 = 0
        high_quality_in_50 = 0

    # Step 3: DIAGNOSIS
    print("\n[3] DIAGNOSIS")
    print("-" * 80)

    if total_count == 0:
        print("❌ PROBLEM: Data does NOT exist in database")
        print("   → This domain is not represented in Polish research (OpenAlex)")
    elif filtered_count == 0:
        print("⚠️  PROBLEM: Data exists, but filters are TOO STRICT")
        print(f"   → {total_count} researchers in field, but 0 meet filter criteria")
        print(f"   → Filters: {filters}")
    elif not is_works_search and high_quality_in_50 == 0:
        print("⚠️  PROBLEM: Data exists, but semantic search CANNOT FIND it")
        print(f"   → {filtered_count} researchers meet criteria")
        print(f"   → But semantic search doesn't return them in top 50")
        print("   → May need even larger candidate pool or query reformulation")
    elif not is_works_search and high_quality_in_50 > 0:
        print("✓ Data EXISTS and semantic search CAN find it")
        print(f"   → {high_quality_in_50} matching results in top 50")
        print("   → Hybrid ranking with candidate_pool_size should work")
    else:
        print(f"✓ Data EXISTS: {filtered_count} items meet criteria")

    return {
        'query_id': query_id,
        'total_count': total_count,
        'filtered_count': filtered_count,
        'semantic_in_50': matching_in_50 if not is_works_search else None,
        'high_quality_in_50': high_quality_in_50 if not is_works_search else None
    }


def main():
    print("="*100)
    print(" INVESTIGATING 5 FAILED VALIDATION QUERIES")
    print("="*100)

    conn = psycopg2.connect(**DB_PARAMS, cursor_factory=RealDictCursor)

    try:
        results = []

        # Q005: Oncologist researching cancer immunotherapy
        results.append(investigate_query(
            conn,
            query_id="Q005",
            query_text="oncologist researching cancer immunotherapy",
            filters={'min_citations': 500},
            concepts_to_check=['oncology', 'cancer', 'immunotherapy', 'tumor']
        ))

        # Q007: Molecular biologist studying CRISPR gene editing
        results.append(investigate_query(
            conn,
            query_id="Q007",
            query_text="molecular biologist studying CRISPR gene editing",
            filters={'min_h_index': 15},
            concepts_to_check=['CRISPR', 'gene editing', 'molecular biology', 'genetic engineering', 'genome editing']
        ))

        # Q012: Cybersecurity and cryptography specialist
        results.append(investigate_query(
            conn,
            query_id="Q012",
            query_text="cybersecurity and cryptography specialist",
            filters={'min_works': 20},
            concepts_to_check=['cybersecurity', 'cryptography', 'computer security', 'information security']
        ))

        # Q014: Renewable energy and photovoltaics (WORKS)
        results.append(investigate_query(
            conn,
            query_id="Q014",
            query_text="renewable energy photovoltaics solar cells",
            filters={'year_from': 2015, 'min_citations': 50},
            concepts_to_check=['renewable energy', 'photovoltaic', 'solar energy', 'solar cell', 'solar power']
        ))

        # Q018: Robotics and AI in industrial automation (WORKS)
        results.append(investigate_query(
            conn,
            query_id="Q018",
            query_text="robotics AI industrial automation manufacturing",
            filters={'year_from': 2016, 'min_citations': 40},
            concepts_to_check=['robotics', 'robot', 'automation', 'industrial automation', 'manufacturing']
        ))

        # Summary
        print("\n\n" + "="*100)
        print(" SUMMARY")
        print("="*100)

        for r in results:
            print(f"\n{r['query_id']}:")
            print(f"  Total in DB: {r['total_count']}")
            print(f"  After filters: {r['filtered_count']}")
            if r['semantic_in_50'] is not None:
                print(f"  Semantic matches (top 50): {r['semantic_in_50']}")
                print(f"  High-quality matches (top 50): {r['high_quality_in_50']}")

        # Save to JSON
        output_file = 'investigation_5_failed_queries.json'
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"\n✓ Results saved to {output_file}")

    finally:
        conn.close()


if __name__ == "__main__":
    main()
