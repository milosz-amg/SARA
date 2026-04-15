#!/usr/bin/env python3
"""
Test with MASSIVE candidate pools to see if we can find high-impact researchers.
"""

from old_data.semantic_search import search_similar_authors

def test_pool_sizes(query, description, target_researchers):
    """Test different pool sizes."""
    print("\n" + "="*100)
    print(f"QUERY: '{query}'")
    print(f"Description: {description}")
    print(f"Looking for: {', '.join(target_researchers)}")
    print("="*100)

    pool_sizes = [500, 1000, 2000, 5000]

    for pool_size in pool_sizes:
        print(f"\n📊 Pool size: {pool_size}")
        try:
            results = search_similar_authors(
                query=query,
                top_k=10,
                boost_metrics=True,
                candidate_pool_size=pool_size,
                min_citations=0,
                min_works=0
            )

            if results:
                print(f"   Results: {len(results)}")
                print(f"   Top 3:")
                for i, r in enumerate(results[:3], 1):
                    print(f"     {i}. {r['display_name']}")
                    print(f"        h={r['h_index']}, cit={r['cited_by_count']}, sim={r['similarity']:.3f}, hybrid={r.get('hybrid_score', 0):.3f}")

                # Check if we found any target researchers
                found_targets = [name for name in target_researchers if any(name.lower() in r['display_name'].lower() for r in results)]
                if found_targets:
                    print(f"\n   ✅ FOUND TARGET: {', '.join(found_targets)}")
                    for name in found_targets:
                        for r in results:
                            if name.lower() in r['display_name'].lower():
                                rank = results.index(r) + 1
                                print(f"      → Rank #{rank}: {r['display_name']} (h={r['h_index']}, cit={r['cited_by_count']})")
                                break
                else:
                    print(f"   ❌ Target researchers NOT in top 10")

        except Exception as e:
            print(f"   ❌ Error: {e}")


def main():
    print("="*100)
    print(" TESTING MASSIVE CANDIDATE POOLS")
    print("="*100)
    print("\nHypothesis: High-impact researchers are far down in semantic ranking")
    print("Solution: Dramatically increase candidate_pool_size (500 → 5000)")

    # Q005: Oncology
    test_pool_sizes(
        query="cancer treatment immunotherapy tumor therapy",
        description="Q005: Oncology & Immunotherapy",
        target_researchers=["Piotr Rutkowski", "Jan Lubiński", "Maciej Wiznerowicz"]
    )

    # Q012: Cybersecurity
    test_pool_sizes(
        query="cryptography encryption network security computer security",
        description="Q012: Cybersecurity",
        target_researchers=["Erol Gelenbe", "Karol Horodecki"]
    )

    # Q007: CRISPR
    test_pool_sizes(
        query="CRISPR gene editing genome modification",
        description="Q007: CRISPR & Gene Editing",
        target_researchers=["Kurt W. Kohn", "Stanisław Krajewski"]
    )

    print("\n" + "="*100)
    print(" ANALYSIS COMPLETE")
    print("="*100)


if __name__ == "__main__":
    main()
