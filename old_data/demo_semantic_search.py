#!/usr/bin/env python3
"""
Demo script for SARA Semantic Search capabilities.
Shows various use cases for finding authors and works.
"""

import json
from old_data.semantic_search import (
    search_similar_authors,
    search_similar_works,
    recommend_collaborators,
    setup_logging
)

def print_separator(title: str = ""):
    """Print a nice separator."""
    print("\n" + "="*80)
    if title:
        print(f"  {title}")
        print("="*80)
    print()

def print_author(author: dict, index: int = None):
    """Pretty print an author result."""
    prefix = f"{index}. " if index else "→ "

    print(f"{prefix}{author['display_name']}")
    print(f"   🏛️  {author.get('last_known_institution_name', 'N/A')}")
    print(f"   📊 {author['works_count']} works, {author['cited_by_count']:,} citations, h-index: {author['h_index']}")
    print(f"   🎯 Similarity: {author['similarity']:.4f}")

    # Show top research areas
    if author.get('concepts'):
        areas = [c.get('display_name', '') for c in author['concepts'][:5] if c.get('display_name')]
        if areas:
            print(f"   🔬 {', '.join(areas)}")

    if author.get('orcid'):
        print(f"   🆔 ORCID: {author['orcid']}")

    print()

def print_work(work: dict, index: int = None):
    """Pretty print a work result."""
    prefix = f"{index}. " if index else "→ "

    print(f"{prefix}{work['title']}")
    print(f"   📅 Year: {work.get('publication_year', 'N/A')} | Type: {work.get('type', 'N/A')}")
    print(f"   📚 Citations: {work.get('cited_by_count', 0):,}")
    print(f"   🎯 Similarity: {work['similarity']:.4f}")

    # Show authors
    if work.get('authorships'):
        authors = [a.get('author', {}).get('display_name', '') for a in work['authorships'][:3]]
        authors = [a for a in authors if a]
        if authors:
            more = f" (+{len(work['authorships']) - 3} more)" if len(work['authorships']) > 3 else ""
            print(f"   👥 {', '.join(authors)}{more}")

    # Show concepts
    if work.get('concepts'):
        concepts = [c.get('display_name', '') for c in work['concepts'][:5] if c.get('display_name')]
        if concepts:
            print(f"   🔬 {', '.join(concepts)}")

    # Open access
    if work.get('open_access_status'):
        oa_emoji = "🔓" if work['open_access_status'] == 'gold' else "🟡"
        print(f"   {oa_emoji} Open Access: {work['open_access_status']}")

    if work.get('doi'):
        print(f"   🔗 DOI: {work['doi']}")

    print()

def demo_1_basic_author_search():
    """Demo 1: Basic author search."""
    print_separator("DEMO 1: Basic Author Search")

    query = "artificial intelligence and machine learning expert"
    print(f"🔍 Query: '{query}'\n")

    results = search_similar_authors(query, top_k=5)

    print(f"📊 Found {len(results)} matching authors:\n")
    for i, author in enumerate(results, 1):
        print_author(author, i)

def demo_2_filtered_author_search():
    """Demo 2: Author search with filters."""
    print_separator("DEMO 2: Filtered Author Search")

    query = "computational biology and bioinformatics"
    print(f"🔍 Query: '{query}'")
    print(f"🎚️  Filters: min_citations=5000, min_works=50\n")

    results = search_similar_authors(
        query,
        top_k=5,
        min_citations=5000,
        min_works=50
    )

    print(f"📊 Found {len(results)} highly cited authors:\n")
    for i, author in enumerate(results, 1):
        print_author(author, i)

def demo_3_institution_search():
    """Demo 3: Search by institution."""
    print_separator("DEMO 3: Search by Institution")

    query = "physics researcher"
    institution = "Warsaw"
    print(f"🔍 Query: '{query}'")
    print(f"🏛️  Institution filter: '{institution}'\n")

    results = search_similar_authors(
        query,
        top_k=5,
        institution=institution
    )

    print(f"📊 Found {len(results)} authors from institutions matching '{institution}':\n")
    for i, author in enumerate(results, 1):
        print_author(author, i)

def demo_4_work_search():
    """Demo 4: Search for research papers."""
    print_separator("DEMO 4: Search for Research Papers")

    query = "quantum computing algorithms and optimization"
    print(f"🔍 Query: '{query}'")
    print(f"🎚️  Filters: year_from=2020, min_citations=50\n")

    results = search_similar_works(
        query,
        top_k=5,
        year_from=2020,
        min_citations=50
    )

    print(f"📊 Found {len(results)} recent papers:\n")
    for i, work in enumerate(results, 1):
        print_work(work, i)

def demo_5_medical_research():
    """Demo 5: Search medical research papers."""
    print_separator("DEMO 5: Medical Research Papers")

    query = "COVID-19 vaccines immunology clinical trials"
    print(f"🔍 Query: '{query}'")
    print(f"🎚️  Filters: year_from=2020, work_type='article'\n")

    results = search_similar_works(
        query,
        top_k=5,
        year_from=2020,
        work_type='article'
    )

    print(f"📊 Found {len(results)} medical articles:\n")
    for i, work in enumerate(results, 1):
        print_work(work, i)

def demo_6_collaborator_recommendation():
    """Demo 6: Recommend collaborators for a project."""
    print_separator("DEMO 6: Collaborator Recommendation")

    project = """
    We are developing a new deep learning framework for analyzing medical images,
    specifically for early detection of cancer. We need experts in computer vision,
    medical imaging, and oncology with strong publication records.
    """

    print(f"📋 Project Description:")
    print(project)
    print(f"\n🎚️  Requirements: h-index >= 15\n")

    results = recommend_collaborators(
        project,
        top_k=5,
        min_h_index=15
    )

    print(f"📊 Top {len(results)} recommended collaborators:\n")
    for i, author in enumerate(results, 1):
        print(f"{i}. {author['display_name']} ({author.get('impact_level', 'N/A')})")
        print(f"   🏛️  {author.get('last_known_institution_name', 'N/A')}")
        print(f"   📊 {author['works_count']} works, {author['cited_by_count']:,} citations, h-index: {author['h_index']}")
        print(f"   🎯 Match Score: {author['similarity']:.4f}")

        if author.get('top_research_areas'):
            print(f"   🔬 Expertise: {', '.join(author['top_research_areas'])}")

        if author.get('orcid'):
            print(f"   🆔 ORCID: {author['orcid']}")

        print()

def demo_7_cross_disciplinary_search():
    """Demo 7: Cross-disciplinary collaboration."""
    print_separator("DEMO 7: Cross-Disciplinary Collaboration")

    project = """
    Interdisciplinary project combining climate science, economics, and policy.
    Looking for researchers with experience in climate modeling, environmental
    economics, or sustainability policy.
    """

    print(f"📋 Project Description:")
    print(project)
    print(f"\n🎚️  Requirements: h-index >= 10\n")

    results = recommend_collaborators(
        project,
        top_k=5,
        min_h_index=10
    )

    print(f"📊 Top {len(results)} interdisciplinary researchers:\n")
    for i, author in enumerate(results, 1):
        print_author(author, i)
        if author.get('impact_level'):
            print(f"   💫 Impact Level: {author['impact_level']}")
            print()

def main():
    """Run all demos."""
    setup_logging()

    print("\n" + "█"*80)
    print("  🚀 SARA - Semantic Search Demonstration")
    print("  " + "─"*76)
    print("  Scientific Assistant for Research and Analysis")
    print("█"*80)

    demos = [
        ("1", "Basic Author Search", demo_1_basic_author_search),
        ("2", "Filtered Author Search", demo_2_filtered_author_search),
        ("3", "Search by Institution", demo_3_institution_search),
        ("4", "Search Research Papers", demo_4_work_search),
        ("5", "Medical Research", demo_5_medical_research),
        ("6", "Collaborator Recommendations", demo_6_collaborator_recommendation),
        ("7", "Cross-Disciplinary Search", demo_7_cross_disciplinary_search),
    ]

    print("\n📋 Available Demos:")
    for num, name, _ in demos:
        print(f"   [{num}] {name}")
    print(f"   [A] Run All Demos")
    print(f"   [Q] Quit")

    choice = input("\n👉 Select a demo (or press Enter for all): ").strip().upper()

    if not choice or choice == "A":
        # Run all demos
        for _, _, demo_func in demos:
            try:
                demo_func()
                input("\n⏸️  Press Enter to continue to next demo...")
            except KeyboardInterrupt:
                print("\n\n👋 Demo interrupted!")
                break
            except Exception as e:
                print(f"\n❌ Error in demo: {e}")
                import traceback
                traceback.print_exc()
    elif choice == "Q":
        print("\n👋 Goodbye!")
        return
    elif choice.isdigit() and 1 <= int(choice) <= len(demos):
        # Run specific demo
        _, _, demo_func = demos[int(choice) - 1]
        try:
            demo_func()
        except Exception as e:
            print(f"\n❌ Error in demo: {e}")
            import traceback
            traceback.print_exc()
    else:
        print("\n❌ Invalid choice!")

    print("\n" + "█"*80)
    print("  ✅ Demo Complete!")
    print("█"*80 + "\n")

if __name__ == "__main__":
    main()
