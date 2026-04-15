#!/usr/bin/env python3
"""
Evaluation script for SARA Semantic Search.
Runs validation queries and analyzes result quality.
"""

import json
import time
from datetime import datetime
from typing import Dict, List, Any
from pathlib import Path

from old_data.semantic_search import (
    search_similar_authors,
    search_similar_works,
    setup_logging
)

# Configuration
VALIDATION_FILE = "validation_queries.json"
RESULTS_DIR = Path("validation_results")
RESULTS_DIR.mkdir(exist_ok=True)

def load_queries() -> Dict:
    """Load validation queries from JSON file."""
    with open(VALIDATION_FILE, 'r', encoding='utf-8') as f:
        return json.load(f)

def run_author_query(query_data: Dict) -> Dict:
    """Run a single author search query."""
    query = query_data.get('query_en', query_data.get('query_pl'))
    expected = query_data.get('expected_characteristics', {})

    # Extract filters from expected characteristics
    min_works = expected.get('min_works', 0)
    min_citations = expected.get('min_citations', 0)
    min_h_index = expected.get('min_h_index', 0)

    # Run search
    start_time = time.time()
    results = search_similar_authors(
        query=query,
        top_k=10,
        min_works=min_works,
        min_citations=min_citations
    )
    elapsed = time.time() - start_time

    # Filter by h-index if specified (post-filter since it's not in search_similar_authors params)
    if min_h_index > 0:
        results = [r for r in results if r.get('h_index', 0) >= min_h_index]

    return {
        'results': results,
        'elapsed_time': elapsed,
        'result_count': len(results)
    }

def run_work_query(query_data: Dict) -> Dict:
    """Run a single work search query."""
    query = query_data.get('query_en', query_data.get('query_pl'))
    expected = query_data.get('expected_characteristics', {})

    # Extract filters
    year_from = expected.get('year_from')
    year_to = expected.get('year_to')
    min_citations = expected.get('min_citations', 0)

    # Run search
    start_time = time.time()
    results = search_similar_works(
        query=query,
        top_k=10,
        year_from=year_from,
        year_to=year_to,
        min_citations=min_citations
    )
    elapsed = time.time() - start_time

    return {
        'results': results,
        'elapsed_time': elapsed,
        'result_count': len(results)
    }

def analyze_author_results(query_data: Dict, search_results: Dict) -> Dict:
    """Analyze quality of author search results."""
    results = search_results['results']
    expected = query_data.get('expected_characteristics', {})
    expected_concepts = expected.get('concepts', [])

    analysis = {
        'has_results': len(results) > 0,
        'result_count': len(results),
        'avg_similarity': 0.0,
        'min_similarity': 0.0,
        'max_similarity': 0.0,
        'concept_match_rate': 0.0,
        'avg_h_index': 0.0,
        'avg_citations': 0.0,
        'top_result': None
    }

    if not results:
        return analysis

    # Calculate statistics
    similarities = [r['similarity'] for r in results]
    analysis['avg_similarity'] = sum(similarities) / len(similarities)
    analysis['min_similarity'] = min(similarities)
    analysis['max_similarity'] = max(similarities)

    h_indices = [r.get('h_index', 0) for r in results]
    analysis['avg_h_index'] = sum(h_indices) / len(h_indices) if h_indices else 0

    citations = [r.get('cited_by_count', 0) for r in results]
    analysis['avg_citations'] = sum(citations) / len(citations) if citations else 0

    # Top result info
    if results:
        top = results[0]
        analysis['top_result'] = {
            'name': top.get('display_name'),
            'institution': top.get('last_known_institution_name'),
            'h_index': top.get('h_index'),
            'citations': top.get('cited_by_count'),
            'similarity': top.get('similarity')
        }

    # Check concept overlap
    if expected_concepts:
        concept_matches = 0
        total_checks = 0

        for result in results:
            result_concepts = result.get('concepts', [])
            if isinstance(result_concepts, list):
                result_concept_names = [
                    c.get('display_name', '').lower()
                    for c in result_concepts[:10]
                ]

                for exp_concept in expected_concepts:
                    total_checks += 1
                    if any(exp_concept.lower() in rc for rc in result_concept_names):
                        concept_matches += 1

        analysis['concept_match_rate'] = concept_matches / total_checks if total_checks > 0 else 0

    return analysis

def analyze_work_results(query_data: Dict, search_results: Dict) -> Dict:
    """Analyze quality of work search results."""
    results = search_results['results']
    expected = query_data.get('expected_characteristics', {})
    expected_concepts = expected.get('concepts', [])

    analysis = {
        'has_results': len(results) > 0,
        'result_count': len(results),
        'avg_similarity': 0.0,
        'min_similarity': 0.0,
        'max_similarity': 0.0,
        'concept_match_rate': 0.0,
        'avg_citations': 0.0,
        'avg_year': 0.0,
        'top_result': None
    }

    if not results:
        return analysis

    # Calculate statistics
    similarities = [r['similarity'] for r in results]
    analysis['avg_similarity'] = sum(similarities) / len(similarities)
    analysis['min_similarity'] = min(similarities)
    analysis['max_similarity'] = max(similarities)

    citations = [r.get('cited_by_count', 0) for r in results]
    analysis['avg_citations'] = sum(citations) / len(citations) if citations else 0

    years = [r.get('publication_year', 0) for r in results if r.get('publication_year')]
    analysis['avg_year'] = sum(years) / len(years) if years else 0

    # Top result info
    if results:
        top = results[0]
        analysis['top_result'] = {
            'title': top.get('title'),
            'year': top.get('publication_year'),
            'citations': top.get('cited_by_count'),
            'similarity': top.get('similarity'),
            'type': top.get('type')
        }

    # Check concept overlap
    if expected_concepts:
        concept_matches = 0
        total_checks = 0

        for result in results:
            result_concepts = result.get('concepts', [])
            if isinstance(result_concepts, list):
                result_concept_names = [
                    c.get('display_name', '').lower()
                    for c in result_concepts[:10]
                ]

                for exp_concept in expected_concepts:
                    total_checks += 1
                    if any(exp_concept.lower() in rc for rc in result_concept_names):
                        concept_matches += 1

        analysis['concept_match_rate'] = concept_matches / total_checks if total_checks > 0 else 0

    return analysis

def run_evaluation(save_results: bool = True, verbose: bool = True) -> Dict:
    """Run complete evaluation on all validation queries."""
    print("\n" + "="*80)
    print("  🔍 SARA SEMANTIC SEARCH EVALUATION")
    print("="*80 + "\n")

    # Load queries
    data = load_queries()
    queries = data['queries']

    print(f"📊 Loaded {len(queries)} validation queries")
    print(f"🎯 Target: {data['metadata']['target_queries']} queries\n")

    # Run evaluation
    results = {
        'metadata': {
            'timestamp': datetime.now().isoformat(),
            'total_queries': len(queries),
            'version': data['metadata']['version']
        },
        'queries': [],
        'summary': {
            'total': len(queries),
            'authors': 0,
            'works': 0,
            'avg_elapsed': 0.0,
            'avg_similarity': 0.0,
            'avg_results': 0.0
        }
    }

    total_elapsed = 0.0
    total_similarity = 0.0
    total_results = 0

    # Process each query
    for i, query_data in enumerate(queries, 1):
        query_id = query_data['id']
        query_type = query_data['search_type']
        query_text = query_data.get('query_en', query_data.get('query_pl'))

        if verbose:
            print(f"[{i}/{len(queries)}] Running {query_id}: {query_text[:60]}...")

        try:
            # Run search
            if query_type == 'authors':
                search_results = run_author_query(query_data)
                analysis = analyze_author_results(query_data, search_results)
                results['summary']['authors'] += 1
            else:  # works
                search_results = run_work_query(query_data)
                analysis = analyze_work_results(query_data, search_results)
                results['summary']['works'] += 1

            # Store results
            query_result = {
                'query_id': query_id,
                'query_text': query_text,
                'category': query_data['category'],
                'search_type': query_type,
                'elapsed_time': search_results['elapsed_time'],
                'analysis': analysis,
                'notes': query_data.get('notes', '')
            }

            # Add top 3 results for inspection
            if search_results['results']:
                query_result['top_3_results'] = []
                for r in search_results['results'][:3]:
                    if query_type == 'authors':
                        query_result['top_3_results'].append({
                            'name': r.get('display_name'),
                            'similarity': r.get('similarity'),
                            'h_index': r.get('h_index'),
                            'citations': r.get('cited_by_count')
                        })
                    else:
                        query_result['top_3_results'].append({
                            'title': r.get('title', '')[:100],
                            'similarity': r.get('similarity'),
                            'year': r.get('publication_year'),
                            'citations': r.get('cited_by_count')
                        })

            results['queries'].append(query_result)

            # Update running totals
            total_elapsed += search_results['elapsed_time']
            total_similarity += analysis['avg_similarity']
            total_results += analysis['result_count']

            if verbose:
                print(f"    ✓ {analysis['result_count']} results, "
                      f"avg similarity: {analysis['avg_similarity']:.3f}, "
                      f"time: {search_results['elapsed_time']:.2f}s")
                if analysis.get('top_result'):
                    top = analysis['top_result']
                    if query_type == 'authors':
                        print(f"      Top: {top.get('name')} (h={top.get('h_index')})")
                    else:
                        print(f"      Top: {top.get('title', '')[:60]}...")
                print()

        except Exception as e:
            print(f"    ❌ ERROR: {e}\n")
            results['queries'].append({
                'query_id': query_id,
                'query_text': query_text,
                'category': query_data['category'],
                'search_type': query_type,
                'error': str(e)
            })

    # Calculate summary statistics
    if len(queries) > 0:
        results['summary']['avg_elapsed'] = total_elapsed / len(queries)
        results['summary']['avg_similarity'] = total_similarity / len(queries)
        results['summary']['avg_results'] = total_results / len(queries)

    # Save results
    if save_results:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = RESULTS_DIR / f"evaluation_{timestamp}.json"

        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)

        print(f"\n💾 Results saved to: {output_file}")

    # Print summary
    print("\n" + "="*80)
    print("  📊 EVALUATION SUMMARY")
    print("="*80)
    print(f"Total queries:        {results['summary']['total']}")
    print(f"  - Author searches:  {results['summary']['authors']}")
    print(f"  - Work searches:    {results['summary']['works']}")
    print(f"Avg elapsed time:     {results['summary']['avg_elapsed']:.2f}s")
    print(f"Avg similarity score: {results['summary']['avg_similarity']:.3f}")
    print(f"Avg results per query: {results['summary']['avg_results']:.1f}")
    print("="*80 + "\n")

    return results

def generate_report(results: Dict) -> str:
    """Generate a human-readable report from evaluation results."""
    report = []
    report.append("# SARA Semantic Search - Validation Report")
    report.append(f"\n**Generated:** {results['metadata']['timestamp']}")
    report.append(f"**Total Queries:** {results['metadata']['total_queries']}\n")

    report.append("## Summary Statistics\n")
    summary = results['summary']
    report.append(f"- **Author Searches:** {summary['authors']}")
    report.append(f"- **Work Searches:** {summary['works']}")
    report.append(f"- **Average Query Time:** {summary['avg_elapsed']:.2f}s")
    report.append(f"- **Average Similarity:** {summary['avg_similarity']:.3f}")
    report.append(f"- **Average Results:** {summary['avg_results']:.1f}\n")

    # Group by category
    by_category = {}
    for q in results['queries']:
        cat = q['category']
        if cat not in by_category:
            by_category[cat] = []
        by_category[cat].append(q)

    report.append("## Results by Category\n")
    for category, queries in sorted(by_category.items()):
        report.append(f"### {category.replace('_', ' ').title()}\n")

        for q in queries:
            report.append(f"**{q['query_id']}:** {q['query_text']}")

            if 'error' in q:
                report.append(f"  - ❌ ERROR: {q['error']}\n")
                continue

            analysis = q['analysis']
            report.append(f"  - Results: {analysis['result_count']}")
            report.append(f"  - Avg Similarity: {analysis['avg_similarity']:.3f}")
            report.append(f"  - Concept Match: {analysis.get('concept_match_rate', 0):.1%}")

            if q.get('top_3_results'):
                report.append(f"  - **Top Result:**")
                top = q['top_3_results'][0]
                if q['search_type'] == 'authors':
                    report.append(f"    - {top['name']} (h-index: {top['h_index']}, similarity: {top['similarity']:.3f})")
                else:
                    report.append(f"    - {top['title']} (year: {top['year']}, similarity: {top['similarity']:.3f})")

            report.append("")

    return "\n".join(report)

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Evaluate SARA semantic search quality')
    parser.add_argument('--no-save', action='store_true', help='Do not save results to file')
    parser.add_argument('--quiet', action='store_true', help='Minimal output')
    parser.add_argument('--report', action='store_true', help='Generate markdown report')

    args = parser.parse_args()

    setup_logging()

    # Run evaluation
    results = run_evaluation(
        save_results=not args.no_save,
        verbose=not args.quiet
    )

    # Generate report if requested
    if args.report:
        report = generate_report(results)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_file = RESULTS_DIR / f"report_{timestamp}.md"

        with open(report_file, 'w', encoding='utf-8') as f:
            f.write(report)

        print(f"📄 Report saved to: {report_file}")
