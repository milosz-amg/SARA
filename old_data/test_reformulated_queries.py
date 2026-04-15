#!/usr/bin/env python3
"""
Test reformulated queries for Q005, Q007, Q012.
Compare profession-based vs research-area-based queries.
"""

from old_data.semantic_search import search_similar_authors
import json

def test_query_comparison(query_id, original_query, reformulated_query, filters):
    """Test original vs reformulated query."""
    print("\n" + "="*100)
    print(f"{query_id}")
    print("="*100)

    # Convert filters to search_similar_authors parameters
    search_filters = {}
    if 'min_citations' in filters:
        search_filters['min_citations'] = filters['min_citations']
    if 'min_works' in filters:
        search_filters['min_works'] = filters['min_works']
    # Note: min_h_index not supported, skip it

    print(f"\n❌ ORIGINAL: '{original_query}'")
    print(f"   Filters: {search_filters}")
    try:
        original_results = search_similar_authors(
            query=original_query,
            top_k=10,
            boost_metrics=True,
            **search_filters
        )
        print(f"   Results: {len(original_results)}")
        if original_results:
            print(f"   Top 3:")
            for i, r in enumerate(original_results[:3], 1):
                print(f"     {i}. {r['display_name']}")
                print(f"        h={r['h_index']}, cit={r['cited_by_count']}, sim={r['similarity']:.3f}, hybrid={r.get('hybrid_score', 0):.3f}")
        else:
            print("   ⚠️  NO RESULTS")
    except Exception as e:
        print(f"   ❌ Error: {e}")
        original_results = []

    print(f"\n✅ REFORMULATED: '{reformulated_query}'")
    print(f"   Filters: {search_filters}")
    try:
        reformulated_results = search_similar_authors(
            query=reformulated_query,
            top_k=10,
            boost_metrics=True,
            **search_filters
        )
        print(f"   Results: {len(reformulated_results)}")
        if reformulated_results:
            print(f"   Top 3:")
            for i, r in enumerate(reformulated_results[:3], 1):
                print(f"     {i}. {r['display_name']}")
                print(f"        h={r['h_index']}, cit={r['cited_by_count']}, sim={r['similarity']:.3f}, hybrid={r.get('hybrid_score', 0):.3f}")
        else:
            print("   ⚠️  NO RESULTS")
    except Exception as e:
        print(f"   ❌ Error: {e}")
        reformulated_results = []

    # Also test WITHOUT filters to see if semantic search works at all
    print(f"\n🔬 REFORMULATED (NO FILTERS): '{reformulated_query}'")
    try:
        no_filter_results = search_similar_authors(
            query=reformulated_query,
            top_k=10,
            boost_metrics=True,
            min_citations=0,
            min_works=0
        )
        print(f"   Results: {len(no_filter_results)}")
        if no_filter_results:
            print(f"   Top 3:")
            for i, r in enumerate(no_filter_results[:3], 1):
                print(f"     {i}. {r['display_name']}")
                print(f"        h={r['h_index']}, cit={r['cited_by_count']}, sim={r['similarity']:.3f}, hybrid={r.get('hybrid_score', 0):.3f}")
    except Exception as e:
        print(f"   ❌ Error: {e}")
        no_filter_results = []

    # Compare
    improvement = len(reformulated_results) - len(original_results)
    if improvement > 0:
        print(f"\n   📈 IMPROVEMENT: +{improvement} results")
    elif improvement < 0:
        print(f"\n   📉 WORSE: {improvement} results")
    else:
        print(f"\n   ➡️  NO CHANGE: {len(reformulated_results)} results")

    return {
        'query_id': query_id,
        'original_count': len(original_results),
        'reformulated_count': len(reformulated_results),
        'improvement': improvement,
        'original_top': original_results[0]['display_name'] if original_results else None,
        'reformulated_top': reformulated_results[0]['display_name'] if reformulated_results else None
    }


def main():
    print("="*100)
    print(" TESTING REFORMULATED QUERIES")
    print("="*100)
    print("\nStrategy: Change from PROFESSION-based to RESEARCH-AREA-based queries")

    results = []

    # Q005: Oncology
    results.append(test_query_comparison(
        query_id="Q005",
        original_query="oncologist researching cancer immunotherapy",
        reformulated_query="cancer treatment immunotherapy tumor therapy",
        filters={'min_citations': 500}
    ))

    # Q007: CRISPR
    results.append(test_query_comparison(
        query_id="Q007",
        original_query="molecular biologist studying CRISPR gene editing",
        reformulated_query="CRISPR gene editing genome modification",
        filters={'min_h_index': 15}
    ))

    # Q012: Cybersecurity
    results.append(test_query_comparison(
        query_id="Q012",
        original_query="cybersecurity and cryptography specialist",
        reformulated_query="cryptography encryption network security computer security",
        filters={'min_works': 20}
    ))

    # Summary
    print("\n\n" + "="*100)
    print(" SUMMARY")
    print("="*100)

    for r in results:
        print(f"\n{r['query_id']}:")
        print(f"  Original:     {r['original_count']} results (top: {r['original_top']})")
        print(f"  Reformulated: {r['reformulated_count']} results (top: {r['reformulated_top']})")
        if r['improvement'] > 0:
            print(f"  ✅ IMPROVED by {r['improvement']} results")
        elif r['improvement'] == 0:
            if r['original_count'] > 0:
                print(f"  ➡️  Same number, but may have different quality")
            else:
                print(f"  ❌ Still 0 results")
        else:
            print(f"  ⚠️  WORSE by {r['improvement']} results")

    # Save results
    output_file = 'reformulated_queries_test.json'
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\n✓ Results saved to {output_file}")


if __name__ == "__main__":
    main()
