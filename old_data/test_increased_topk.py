#!/usr/bin/env python3
"""
Test if increasing top_k helps find high-metric researchers.
"""

from old_data.semantic_search import search_similar_authors, setup_logging

setup_logging()

print("\n" + "="*80)
print("Testing Q001: Cardiology with different top_k values")
print("="*80)

query = "cardiologist specializing in heart failure"

# Test with increasing top_k
for top_k in [10, 50, 100, 500, 1000]:
    print(f"\n🔍 top_k = {top_k}, min_citations=1000, min_h_index=20")

    results = search_similar_authors(
        query=query,
        top_k=top_k,
        min_citations=1000,
        min_works=0,
        institution=None
    )

    print(f"   Results: {len(results)}")

    if results:
        print(f"   Top 3:")
        for i, r in enumerate(results[:3], 1):
            print(f"     {i}. {r['display_name']} (h={r['h_index']}, cit={r['cited_by_count']:,}, sim={r['similarity']:.3f})")

print("\n" + "="*80)
print("Now testing without h_index filter (just citations)")
print("="*80)

for top_k in [10, 50, 100]:
    print(f"\n🔍 top_k = {top_k}, min_citations=1000, NO h_index filter")

    results = search_similar_authors(
        query=query,
        top_k=top_k,
        min_citations=1000,
        min_works=0
    )

    print(f"   Results: {len(results)}")

    if results:
        print(f"   Top 3:")
        for i, r in enumerate(results[:3], 1):
            print(f"     {i}. {r['display_name']} (h={r['h_index']}, cit={r['cited_by_count']:,}, sim={r['similarity']:.3f})")
