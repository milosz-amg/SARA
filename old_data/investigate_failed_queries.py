#!/usr/bin/env python3
"""
Investigate why Q005, Q007, Q012, Q014, Q018 return 0 results.
Check database using SQL queries to understand data availability.
"""

import psycopg2
import json
from typing import Dict, List, Tuple

# Database connection parameters
DB_PARAMS = {
    'host': 'localhost',
    'port': 5432,
    'database': 'devdb',
    'user': 'devuser',
    'password': 'devpass'
}


def connect_db():
    """Connect to PostgreSQL database."""
    return psycopg2.connect(**DB_PARAMS)


def investigate_q005_oncology_immunotherapy(conn) -> Dict:
    """
    Q005: Oncologist researching cancer immunotherapy
    Expected: min_citations=500
    """
    print("\n" + "="*80)
    print("Q005: Oncology & Cancer Immunotherapy")
    print("="*80)

    # Simplified query - use indexed search
    query = """
    SELECT
      COUNT(*) as total_count,
      COUNT(CASE WHEN cited_by_count >= 500 THEN 1 END) as with_500_citations,
      COUNT(CASE WHEN cited_by_count >= 100 THEN 1 END) as with_100_citations,
      COUNT(CASE WHEN cited_by_count >= 50 THEN 1 END) as with_50_citations,
      ROUND(AVG(cited_by_count), 2) as avg_citations,
      MAX(cited_by_count) as max_citations,
      ROUND(AVG(h_index), 2) as avg_h_index,
      MAX(h_index) as max_h_index
    FROM authors
    WHERE concepts::text ILIKE '%oncology%'
       OR concepts::text ILIKE '%cancer%'
       OR concepts::text ILIKE '%immunotherapy%';
    """

    with conn.cursor() as cur:
        cur.execute(query)
        result = cur.fetchone()

        stats = {
            'total_count': result[0],
            'with_500_citations': result[1],
            'with_100_citations': result[2],
            'with_50_citations': result[3],
            'avg_citations': float(result[4]) if result[4] else 0,
            'max_citations': result[5],
            'avg_h_index': float(result[6]) if result[6] else 0,
            'max_h_index': result[7]
        }

        print(f"Total researchers: {stats['total_count']}")
        print(f"With ≥500 citations: {stats['with_500_citations']}")
        print(f"With ≥100 citations: {stats['with_100_citations']}")
        print(f"With ≥50 citations: {stats['with_50_citations']}")
        print(f"Average citations: {stats['avg_citations']:.2f}")
        print(f"Max citations: {stats['max_citations']}")
        print(f"Average h-index: {stats['avg_h_index']:.2f}")
        print(f"Max h-index: {stats['max_h_index']}")

        # Get top researchers
        print("\nTop 5 oncology/cancer researchers:")
        cur.execute("""
            SELECT display_name, cited_by_count, h_index, works_count
            FROM authors
            WHERE (concepts::text ILIKE '%oncology%' OR concepts::text ILIKE '%cancer%')
              AND cited_by_count > 0
            ORDER BY cited_by_count DESC
            LIMIT 5;
        """)
        for row in cur.fetchall():
            print(f"  - {row[0]}: {row[1]} citations, h={row[2]}, {row[3]} works")

    return stats


def investigate_q007_crispr(conn) -> Dict:
    """
    Q007: Molecular biologist studying CRISPR gene editing
    Expected: min_h_index=15
    """
    print("\n" + "="*80)
    print("Q007: CRISPR & Gene Editing")
    print("="*80)

    # Simplified query
    query = """
    SELECT
      COUNT(*) as total_count,
      COUNT(CASE WHEN h_index >= 15 THEN 1 END) as with_h15,
      COUNT(CASE WHEN h_index >= 10 THEN 1 END) as with_h10,
      COUNT(CASE WHEN h_index >= 5 THEN 1 END) as with_h5,
      ROUND(AVG(cited_by_count), 2) as avg_citations,
      MAX(cited_by_count) as max_citations,
      ROUND(AVG(h_index), 2) as avg_h_index,
      MAX(h_index) as max_h_index
    FROM authors
    WHERE concepts::text ILIKE '%crispr%'
       OR concepts::text ILIKE '%gene editing%'
       OR concepts::text ILIKE '%molecular biology%'
       OR concepts::text ILIKE '%genetic engineering%';
    """

    with conn.cursor() as cur:
        cur.execute(query)
        result = cur.fetchone()

        stats = {
            'total_count': result[0],
            'with_h15': result[1],
            'with_h10': result[2],
            'with_h5': result[3],
            'avg_citations': float(result[4]) if result[4] else 0,
            'max_citations': result[5],
            'avg_h_index': float(result[6]) if result[6] else 0,
            'max_h_index': result[7]
        }

        print(f"Total researchers: {stats['total_count']}")
        print(f"With h-index ≥15: {stats['with_h15']}")
        print(f"With h-index ≥10: {stats['with_h10']}")
        print(f"With h-index ≥5: {stats['with_h5']}")
        print(f"Average citations: {stats['avg_citations']:.2f}")
        print(f"Max citations: {stats['max_citations']}")
        print(f"Average h-index: {stats['avg_h_index']:.2f}")
        print(f"Max h-index: {stats['max_h_index']}")

        # Check specifically for CRISPR
        print("\nChecking specifically for 'CRISPR' term:")
        cur.execute("""
            SELECT COUNT(*)
            FROM authors
            WHERE concepts::text ILIKE '%crispr%';
        """)
        crispr_count = cur.fetchone()[0]
        print(f"  Researchers with 'CRISPR' concept: {crispr_count}")

        # Get top researchers
        print("\nTop 5 molecular biology/genetics researchers:")
        cur.execute("""
            SELECT display_name, cited_by_count, h_index, works_count
            FROM authors
            WHERE (concepts::text ILIKE '%molecular biology%'
               OR concepts::text ILIKE '%genetic engineering%')
              AND h_index > 0
            ORDER BY h_index DESC
            LIMIT 5;
        """)
        for row in cur.fetchall():
            print(f"  - {row[0]}: {row[1]} citations, h={row[2]}, {row[3]} works")

    return stats


def investigate_q012_cybersecurity(conn) -> Dict:
    """
    Q012: Cybersecurity and cryptography specialist
    Expected: min_works=20
    """
    print("\n" + "="*80)
    print("Q012: Cybersecurity & Cryptography")
    print("="*80)

    # Simplified query
    query = """
    SELECT
      COUNT(*) as total_count,
      COUNT(CASE WHEN works_count >= 20 THEN 1 END) as with_20_works,
      COUNT(CASE WHEN works_count >= 10 THEN 1 END) as with_10_works,
      COUNT(CASE WHEN works_count >= 5 THEN 1 END) as with_5_works,
      ROUND(AVG(works_count), 2) as avg_works,
      MAX(works_count) as max_works,
      ROUND(AVG(cited_by_count), 2) as avg_citations,
      MAX(cited_by_count) as max_citations,
      ROUND(AVG(h_index), 2) as avg_h_index,
      MAX(h_index) as max_h_index
    FROM authors
    WHERE concepts::text ILIKE '%cybersecurity%'
       OR concepts::text ILIKE '%cryptography%'
       OR concepts::text ILIKE '%computer security%'
       OR concepts::text ILIKE '%information security%';
    """

    with conn.cursor() as cur:
        cur.execute(query)
        result = cur.fetchone()

        stats = {
            'total_count': result[0],
            'with_20_works': result[1],
            'with_10_works': result[2],
            'with_5_works': result[3],
            'avg_works': float(result[4]) if result[4] else 0,
            'max_works': result[5],
            'avg_citations': float(result[6]) if result[6] else 0,
            'max_citations': result[7],
            'avg_h_index': float(result[8]) if result[8] else 0,
            'max_h_index': result[9]
        }

        print(f"Total researchers: {stats['total_count']}")
        print(f"With ≥20 works: {stats['with_20_works']}")
        print(f"With ≥10 works: {stats['with_10_works']}")
        print(f"With ≥5 works: {stats['with_5_works']}")
        print(f"Average works: {stats['avg_works']:.2f}")
        print(f"Max works: {stats['max_works']}")
        print(f"Average citations: {stats['avg_citations']:.2f}")
        print(f"Max citations: {stats['max_citations']}")
        print(f"Average h-index: {stats['avg_h_index']:.2f}")
        print(f"Max h-index: {stats['max_h_index']}")

        # Get top researchers
        print("\nTop 5 cybersecurity/cryptography researchers:")
        cur.execute("""
            SELECT display_name, cited_by_count, h_index, works_count
            FROM authors
            WHERE (concepts::text ILIKE '%cryptography%' OR concepts::text ILIKE '%computer security%')
              AND h_index > 0
            ORDER BY h_index DESC
            LIMIT 5;
        """)
        for row in cur.fetchall():
            print(f"  - {row[0]}: {row[1]} citations, h={row[2]}, {row[3]} works")

    return stats


def investigate_q014_renewable_energy(conn) -> Dict:
    """
    Q014: Renewable energy and photovoltaics (WORKS search)
    Expected: year_from=2015, min_citations=50
    """
    print("\n" + "="*80)
    print("Q014: Renewable Energy & Photovoltaics (WORKS)")
    print("="*80)

    # Simplified query
    query = """
    SELECT
      COUNT(*) as total_count,
      COUNT(CASE WHEN cited_by_count >= 50 THEN 1 END) as with_50_citations,
      COUNT(CASE WHEN cited_by_count >= 20 THEN 1 END) as with_20_citations,
      COUNT(CASE WHEN cited_by_count >= 10 THEN 1 END) as with_10_citations,
      ROUND(AVG(cited_by_count), 2) as avg_citations,
      MAX(cited_by_count) as max_citations,
      MIN(publication_year) as earliest_year,
      MAX(publication_year) as latest_year
    FROM works
    WHERE publication_year >= 2015
      AND (concepts::text ILIKE '%renewable energy%'
        OR concepts::text ILIKE '%photovoltaic%'
        OR concepts::text ILIKE '%solar energy%'
        OR concepts::text ILIKE '%solar cell%');
    """

    with conn.cursor() as cur:
        cur.execute(query)
        result = cur.fetchone()

        stats = {
            'total_count': result[0],
            'with_50_citations': result[1],
            'with_20_citations': result[2],
            'with_10_citations': result[3],
            'avg_citations': float(result[4]) if result[4] else 0,
            'max_citations': result[5],
            'earliest_year': result[6],
            'latest_year': result[7]
        }

        print(f"Total works (2015+): {stats['total_count']}")
        print(f"With ≥50 citations: {stats['with_50_citations']}")
        print(f"With ≥20 citations: {stats['with_20_citations']}")
        print(f"With ≥10 citations: {stats['with_10_citations']}")
        print(f"Average citations: {stats['avg_citations']:.2f}")
        print(f"Max citations: {stats['max_citations']}")
        print(f"Year range: {stats['earliest_year']}-{stats['latest_year']}")

        # Get top works
        print("\nTop 5 renewable energy/photovoltaic works:")
        cur.execute("""
            SELECT title, publication_year, cited_by_count
            FROM works
            WHERE publication_year >= 2015
              AND (concepts::text ILIKE '%photovoltaic%' OR concepts::text ILIKE '%solar energy%')
              AND cited_by_count > 0
            ORDER BY cited_by_count DESC
            LIMIT 5;
        """)
        for row in cur.fetchall():
            title = row[0][:80] + "..." if len(row[0]) > 80 else row[0]
            print(f"  - {title} ({row[1]}): {row[2]} citations")

    return stats


def investigate_q018_robotics(conn) -> Dict:
    """
    Q018: Robotics and AI in industrial automation (WORKS search)
    Expected: year_from=2016, min_citations=40
    """
    print("\n" + "="*80)
    print("Q018: Robotics & Industrial Automation (WORKS)")
    print("="*80)

    # Simplified query
    query = """
    SELECT
      COUNT(*) as total_count,
      COUNT(CASE WHEN cited_by_count >= 40 THEN 1 END) as with_40_citations,
      COUNT(CASE WHEN cited_by_count >= 20 THEN 1 END) as with_20_citations,
      COUNT(CASE WHEN cited_by_count >= 10 THEN 1 END) as with_10_citations,
      ROUND(AVG(cited_by_count), 2) as avg_citations,
      MAX(cited_by_count) as max_citations,
      MIN(publication_year) as earliest_year,
      MAX(publication_year) as latest_year
    FROM works
    WHERE publication_year >= 2016
      AND (concepts::text ILIKE '%robotics%'
        OR concepts::text ILIKE '%robot%'
        OR concepts::text ILIKE '%automation%'
        OR concepts::text ILIKE '%manufacturing%');
    """

    with conn.cursor() as cur:
        cur.execute(query)
        result = cur.fetchone()

        stats = {
            'total_count': result[0],
            'with_40_citations': result[1],
            'with_20_citations': result[2],
            'with_10_citations': result[3],
            'avg_citations': float(result[4]) if result[4] else 0,
            'max_citations': result[5],
            'earliest_year': result[6],
            'latest_year': result[7]
        }

        print(f"Total works (2016+): {stats['total_count']}")
        print(f"With ≥40 citations: {stats['with_40_citations']}")
        print(f"With ≥20 citations: {stats['with_20_citations']}")
        print(f"With ≥10 citations: {stats['with_10_citations']}")
        print(f"Average citations: {stats['avg_citations']:.2f}")
        print(f"Max citations: {stats['max_citations']}")
        print(f"Year range: {stats['earliest_year']}-{stats['latest_year']}")

        # Get top works
        print("\nTop 5 robotics/automation works:")
        cur.execute("""
            SELECT title, publication_year, cited_by_count
            FROM works
            WHERE publication_year >= 2016
              AND (concepts::text ILIKE '%robotics%' OR concepts::text ILIKE '%automation%')
              AND cited_by_count > 0
            ORDER BY cited_by_count DESC
            LIMIT 5;
        """)
        for row in cur.fetchall():
            title = row[0][:80] + "..." if len(row[0]) > 80 else row[0]
            print(f"  - {title} ({row[1]}): {row[2]} citations")

    return stats


def main():
    """Run all investigations."""
    print("="*80)
    print("INVESTIGATING FAILED VALIDATION QUERIES")
    print("="*80)
    print("\nConnecting to database...")

    conn = connect_db()

    try:
        results = {
            'Q005': investigate_q005_oncology_immunotherapy(conn),
            'Q007': investigate_q007_crispr(conn),
            'Q012': investigate_q012_cybersecurity(conn),
            'Q014': investigate_q014_renewable_energy(conn),
            'Q018': investigate_q018_robotics(conn)
        }

        # Summary
        print("\n" + "="*80)
        print("SUMMARY")
        print("="*80)

        print("\nQ005 (Oncology, min_citations=500):")
        print(f"  - {results['Q005']['with_500_citations']} researchers meet criteria")
        print(f"  - {results['Q005']['total_count']} total in field")

        print("\nQ007 (CRISPR, min_h_index=15):")
        print(f"  - {results['Q007']['with_h15']} researchers meet criteria")
        print(f"  - {results['Q007']['total_count']} total in field")

        print("\nQ012 (Cybersecurity, min_works=20):")
        print(f"  - {results['Q012']['with_20_works']} researchers meet criteria")
        print(f"  - {results['Q012']['total_count']} total in field")

        print("\nQ014 (Renewable Energy works 2015+, min_citations=50):")
        print(f"  - {results['Q014']['with_50_citations']} works meet criteria")
        print(f"  - {results['Q014']['total_count']} total works in field")

        print("\nQ018 (Robotics works 2016+, min_citations=40):")
        print(f"  - {results['Q018']['with_40_citations']} works meet criteria")
        print(f"  - {results['Q018']['total_count']} total works in field")

        # Save results to JSON
        output_file = 'failed_queries_investigation.json'
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"\n✓ Results saved to {output_file}")

    finally:
        conn.close()
        print("\n✓ Database connection closed")


if __name__ == "__main__":
    main()
