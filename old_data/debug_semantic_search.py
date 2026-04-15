#!/usr/bin/env python3
"""
Debug script to investigate why semantic search fails for certain queries.
"""

import psycopg2
from psycopg2.extras import RealDictCursor
from old_data.semantic_search import (
    search_similar_authors,
    get_query_embedding,
    setup_logging
)

DB_PARAMS = {
    'host': 'localhost',
    'port': 5432,
    'database': 'devdb',
    'user': 'devuser',
    'password': 'devpass'
}

def test_query_variations(base_query: str, variations: list):
    """Test multiple variations of a query to see which works."""
    print("\n" + "="*80)
    print(f"🔍 TESTING QUERY VARIATIONS")
    print(f"Base query: '{base_query}'")
    print("="*80 + "\n")

    results_summary = []

    for i, (query, filters) in enumerate(variations, 1):
        print(f"\n[{i}/{len(variations)}] Testing: '{query}'")
        print(f"            Filters: {filters}")

        try:
            results = search_similar_authors(
                query=query,
                top_k=10,
                **filters
            )

            print(f"    ✓ Results: {len(results)}")
            if results:
                top = results[0]
                print(f"    → Top: {top['display_name']} (h={top['h_index']}, cit={top['cited_by_count']}, sim={top['similarity']:.3f})")

            results_summary.append({
                'query': query,
                'filters': filters,
                'count': len(results),
                'top_similarity': results[0]['similarity'] if results else 0
            })

        except Exception as e:
            print(f"    ❌ Error: {e}")
            results_summary.append({
                'query': query,
                'filters': filters,
                'count': 0,
                'error': str(e)
            })

    return results_summary

def check_filter_order(query: str, target_filters: dict):
    """Check if filters are being applied before or after semantic search."""
    print("\n" + "="*80)
    print(f"🔬 FILTER ORDER INVESTIGATION")
    print(f"Query: '{query}'")
    print("="*80 + "\n")

    # Step 1: Get embedding
    print("Step 1: Getting query embedding...")
    embedding = get_query_embedding(query)
    print(f"  ✓ Embedding: {len(embedding)} dimensions")
    print(f"  ✓ First 5 values: {embedding[:5]}")

    # Step 2: Check raw SQL without filters
    conn = psycopg2.connect(**DB_PARAMS, cursor_factory=RealDictCursor)
    cursor = conn.cursor()

    embedding_str = str(embedding)

    print("\nStep 2: Semantic search WITHOUT filters...")
    sql_no_filters = f"""
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

    cursor.execute(sql_no_filters, [embedding_str, embedding_str])
    results_no_filters = cursor.fetchall()

    print(f"  ✓ Found {len(results_no_filters)} results")
    for i, r in enumerate(results_no_filters[:5], 1):
        print(f"    {i}. {r['display_name']} (h={r['h_index']}, cit={r['cited_by_count']}, sim={r['similarity']:.3f})")

    # Step 3: Apply filters AFTER semantic search
    print(f"\nStep 3: Applying filters POST-search (min_citations={target_filters.get('min_citations', 0)}, min_h_index={target_filters.get('min_h_index', 0)})...")

    filtered_results = [
        r for r in results_no_filters
        if r['cited_by_count'] >= target_filters.get('min_citations', 0)
        and r['h_index'] >= target_filters.get('min_h_index', 0)
    ]

    print(f"  ✓ After filtering: {len(filtered_results)} results")
    for i, r in enumerate(filtered_results[:5], 1):
        print(f"    {i}. {r['display_name']} (h={r['h_index']}, cit={r['cited_by_count']}, sim={r['similarity']:.3f})")

    # Step 4: Apply filters IN SQL (how our function does it)
    print(f"\nStep 4: Filters applied IN SQL (as in our function)...")

    filters_sql = ["embedding IS NOT NULL"]
    params = [embedding_str]

    if target_filters.get('min_citations', 0) > 0:
        filters_sql.append("cited_by_count >= %s")
        params.append(target_filters['min_citations'])

    if target_filters.get('min_h_index', 0) > 0:
        filters_sql.append("h_index >= %s")
        params.append(target_filters['min_h_index'])

    where_clause = " AND ".join(filters_sql)
    params_final = [embedding_str] + params[1:] + [embedding_str, 10]

    sql_with_filters = f"""
        SELECT
            display_name,
            h_index,
            cited_by_count,
            works_count,
            1 - (embedding <=> %s::vector) as similarity
        FROM authors
        WHERE {where_clause}
        ORDER BY embedding <=> %s::vector
        LIMIT %s;
    """

    print(f"  SQL WHERE: {where_clause}")
    print(f"  Params: {params_final}")

    cursor.execute(sql_with_filters, params_final)
    results_with_filters = cursor.fetchall()

    print(f"  ✓ Found {len(results_with_filters)} results")
    for i, r in enumerate(results_with_filters[:5], 1):
        print(f"    {i}. {r['display_name']} (h={r['h_index']}, cit={r['cited_by_count']}, sim={r['similarity']:.3f})")

    cursor.close()
    conn.close()

    return {
        'no_filters': len(results_no_filters),
        'post_filtered': len(filtered_results),
        'sql_filtered': len(results_with_filters)
    }

def check_concept_coverage(query: str, expected_concepts: list):
    """Check how many authors in DB have the expected concepts."""
    print("\n" + "="*80)
    print(f"📚 CONCEPT COVERAGE CHECK")
    print(f"Query: '{query}'")
    print(f"Expected concepts: {expected_concepts}")
    print("="*80 + "\n")

    conn = psycopg2.connect(**DB_PARAMS, cursor_factory=RealDictCursor)
    cursor = conn.cursor()

    for concept in expected_concepts:
        sql = f"""
            SELECT COUNT(*) as count,
                   MIN(h_index) as min_h,
                   MAX(h_index) as max_h,
                   ROUND(AVG(h_index), 1) as avg_h,
                   MIN(cited_by_count) as min_cit,
                   MAX(cited_by_count) as max_cit
            FROM authors
            WHERE concepts::text ILIKE %s;
        """

        cursor.execute(sql, [f'%{concept}%'])
        result = cursor.fetchone()

        print(f"\nConcept: '{concept}'")
        print(f"  Total authors: {result['count']}")
        print(f"  H-index range: {result['min_h']} - {result['max_h']} (avg: {result['avg_h']})")
        print(f"  Citations range: {result['min_cit']} - {result['max_cit']}")

    cursor.close()
    conn.close()

def main():
    """Run debugging tests."""
    setup_logging()

    print("\n" + "█"*80)
    print("  🐛 SEMANTIC SEARCH DEBUGGER")
    print("█"*80)

    # Test Case 1: Q001 - Cardiology
    print("\n\n" + "█"*80)
    print("  TEST CASE 1: Q001 - CARDIOLOGY")
    print("█"*80)

    cardiology_variations = [
        ("cardiologist specializing in heart failure", {'min_citations': 1000, 'min_h_index': 20}),
        ("cardiologist specializing in heart failure", {'min_citations': 500, 'min_h_index': 10}),
        ("cardiologist specializing in heart failure", {'min_citations': 0, 'min_h_index': 0}),
        ("cardiologist heart failure", {'min_citations': 0, 'min_h_index': 0}),
        ("cardiology researcher", {'min_citations': 0, 'min_h_index': 0}),
        ("cardiology", {'min_citations': 0, 'min_h_index': 0}),
    ]

    test_query_variations("cardiologist specializing in heart failure", cardiology_variations)

    # Deep dive on filter order
    check_filter_order(
        "cardiologist specializing in heart failure",
        {'min_citations': 1000, 'min_h_index': 20}
    )

    # Check concept coverage
    check_concept_coverage(
        "cardiologist specializing in heart failure",
        ["Cardiology", "Heart Failure", "Internal Medicine"]
    )

    # Test Case 2: Q005 - Oncology
    print("\n\n" + "█"*80)
    print("  TEST CASE 2: Q005 - ONCOLOGY IMMUNOTHERAPY")
    print("█"*80)

    oncology_variations = [
        ("oncologist researching cancer immunotherapy", {'min_citations': 500, 'min_h_index': 0}),
        ("oncologist researching cancer immunotherapy", {'min_citations': 100, 'min_h_index': 0}),
        ("oncologist researching cancer immunotherapy", {'min_citations': 0, 'min_h_index': 0}),
        ("cancer immunotherapy researcher", {'min_citations': 0, 'min_h_index': 0}),
        ("oncology researcher", {'min_citations': 0, 'min_h_index': 0}),
        ("oncology", {'min_citations': 0, 'min_h_index': 0}),
    ]

    test_query_variations("oncologist researching cancer immunotherapy", oncology_variations)

    check_filter_order(
        "oncologist researching cancer immunotherapy",
        {'min_citations': 500, 'min_h_index': 0}
    )

    check_concept_coverage(
        "oncologist researching cancer immunotherapy",
        ["Oncology", "Immunotherapy", "Cancer Research"]
    )

    # Test Case 3: Known good query for comparison
    print("\n\n" + "█"*80)
    print("  TEST CASE 3: CONTROL - MACHINE LEARNING (Known to work)")
    print("█"*80)

    ml_variations = [
        ("machine learning expert", {'min_works': 20, 'min_citations': 0}),
        ("machine learning expert", {'min_works': 0, 'min_citations': 0}),
        ("machine learning", {'min_works': 0, 'min_citations': 0}),
    ]

    test_query_variations("machine learning expert", ml_variations)

    print("\n\n" + "█"*80)
    print("  ✅ DEBUGGING COMPLETE")
    print("█"*80 + "\n")

if __name__ == "__main__":
    main()
