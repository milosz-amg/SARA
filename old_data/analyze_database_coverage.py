#!/usr/bin/env python3
"""
Database Coverage Analysis for SARA
Analyzes what data is actually available in the database to inform validation query design.
"""

import psycopg2
from psycopg2.extras import RealDictCursor
import json
from datetime import datetime

DB_PARAMS = {
    'host': 'localhost',
    'port': 5432,
    'database': 'devdb',
    'user': 'devuser',
    'password': 'devpass'
}

def run_query(sql: str, description: str):
    """Execute a SQL query and return results."""
    print(f"\n{'='*80}")
    print(f"📊 {description}")
    print('='*80)
    print(f"\nSQL:\n{sql}\n")

    conn = psycopg2.connect(**DB_PARAMS, cursor_factory=RealDictCursor)
    cursor = conn.cursor()

    try:
        cursor.execute(sql)
        results = cursor.fetchall()

        if results:
            # Print results in a nice table format
            if len(results) > 0:
                # Get column names
                cols = results[0].keys()

                # Print header
                header = " | ".join(str(col).ljust(20) for col in cols)
                print(header)
                print("-" * len(header))

                # Print rows
                for row in results:
                    print(" | ".join(str(row[col])[:20].ljust(20) for col in cols))
        else:
            print("No results returned.")

        print(f"\n✓ Returned {len(results)} row(s)")

        cursor.close()
        conn.close()

        return results

    except Exception as e:
        print(f"❌ Error: {e}")
        cursor.close()
        conn.close()
        return []

def analyze_h_index_distribution():
    """Analyze h-index distribution in the database."""
    print("\n" + "█"*80)
    print("  H-INDEX DISTRIBUTION ANALYSIS")
    print("█"*80)

    sql = """
    SELECT
      CASE
        WHEN h_index = 0 THEN '0'
        WHEN h_index BETWEEN 1 AND 5 THEN '1-5'
        WHEN h_index BETWEEN 6 AND 10 THEN '6-10'
        WHEN h_index BETWEEN 11 AND 20 THEN '11-20'
        WHEN h_index BETWEEN 21 AND 30 THEN '21-30'
        WHEN h_index BETWEEN 31 AND 50 THEN '31-50'
        WHEN h_index BETWEEN 51 AND 100 THEN '51-100'
        WHEN h_index > 100 THEN '100+'
      END as h_index_range,
      COUNT(*) as count,
      ROUND(100.0 * COUNT(*) / SUM(COUNT(*)) OVER (), 2) as percentage
    FROM authors
    GROUP BY h_index_range
    ORDER BY MIN(h_index);
    """

    results = run_query(sql, "H-Index Distribution by Range")

    # Threshold analysis
    sql2 = """
    SELECT
      COUNT(CASE WHEN h_index >= 5 THEN 1 END) as h_gte_5,
      COUNT(CASE WHEN h_index >= 10 THEN 1 END) as h_gte_10,
      COUNT(CASE WHEN h_index >= 15 THEN 1 END) as h_gte_15,
      COUNT(CASE WHEN h_index >= 20 THEN 1 END) as h_gte_20,
      COUNT(CASE WHEN h_index >= 30 THEN 1 END) as h_gte_30,
      COUNT(CASE WHEN h_index >= 50 THEN 1 END) as h_gte_50,
      COUNT(CASE WHEN h_index >= 100 THEN 1 END) as h_gte_100,
      COUNT(*) as total
    FROM authors;
    """

    results2 = run_query(sql2, "Authors Meeting H-Index Thresholds")

    return results, results2

def analyze_citation_distribution():
    """Analyze citation count distribution."""
    print("\n" + "█"*80)
    print("  CITATION DISTRIBUTION ANALYSIS")
    print("█"*80)

    sql = """
    SELECT
      CASE
        WHEN cited_by_count = 0 THEN '0'
        WHEN cited_by_count BETWEEN 1 AND 100 THEN '1-100'
        WHEN cited_by_count BETWEEN 101 AND 500 THEN '101-500'
        WHEN cited_by_count BETWEEN 501 AND 1000 THEN '501-1000'
        WHEN cited_by_count BETWEEN 1001 AND 5000 THEN '1001-5000'
        WHEN cited_by_count BETWEEN 5001 AND 10000 THEN '5001-10000'
        WHEN cited_by_count BETWEEN 10001 AND 50000 THEN '10001-50000'
        WHEN cited_by_count > 50000 THEN '50000+'
      END as citations_range,
      COUNT(*) as count,
      ROUND(100.0 * COUNT(*) / SUM(COUNT(*)) OVER (), 2) as percentage
    FROM authors
    GROUP BY citations_range
    ORDER BY MIN(cited_by_count);
    """

    results = run_query(sql, "Citation Distribution by Range")

    # Threshold analysis
    sql2 = """
    SELECT
      COUNT(CASE WHEN cited_by_count >= 100 THEN 1 END) as cit_gte_100,
      COUNT(CASE WHEN cited_by_count >= 500 THEN 1 END) as cit_gte_500,
      COUNT(CASE WHEN cited_by_count >= 1000 THEN 1 END) as cit_gte_1000,
      COUNT(CASE WHEN cited_by_count >= 5000 THEN 1 END) as cit_gte_5000,
      COUNT(CASE WHEN cited_by_count >= 10000 THEN 1 END) as cit_gte_10000,
      COUNT(*) as total
    FROM authors;
    """

    results2 = run_query(sql2, "Authors Meeting Citation Thresholds")

    return results, results2

def analyze_cardiology():
    """Q001: Cardiology with high requirements."""
    print("\n" + "█"*80)
    print("  Q001: CARDIOLOGY ANALYSIS")
    print("█"*80)

    # Count cardiologists
    sql1 = """
    SELECT COUNT(*) as total_cardiologists,
           MIN(h_index) as min_h,
           MAX(h_index) as max_h,
           ROUND(AVG(h_index), 1) as avg_h,
           MIN(cited_by_count) as min_cit,
           MAX(cited_by_count) as max_cit,
           ROUND(AVG(cited_by_count), 0) as avg_cit
    FROM authors
    WHERE concepts::text ILIKE '%cardio%';
    """

    results1 = run_query(sql1, "Cardiology Researchers - Statistics")

    # Top cardiologists
    sql2 = """
    SELECT display_name,
           h_index,
           cited_by_count,
           works_count,
           last_known_institution_name
    FROM authors
    WHERE concepts::text ILIKE '%cardio%'
    ORDER BY cited_by_count DESC
    LIMIT 10;
    """

    results2 = run_query(sql2, "Top 10 Cardiologists by Citations")

    # Check with strict requirements
    sql3 = """
    SELECT COUNT(*) as count_strict
    FROM authors
    WHERE concepts::text ILIKE '%cardio%'
      AND cited_by_count >= 1000
      AND h_index >= 20;
    """

    results3 = run_query(sql3, "Cardiologists Meeting Strict Requirements (cit>=1000, h>=20)")

    # Check with relaxed requirements
    sql4 = """
    SELECT display_name, h_index, cited_by_count, works_count
    FROM authors
    WHERE concepts::text ILIKE '%cardio%'
      AND cited_by_count >= 500
      AND h_index >= 10
    ORDER BY cited_by_count DESC
    LIMIT 10;
    """

    results4 = run_query(sql4, "Cardiologists with Relaxed Requirements (cit>=500, h>=10)")

    return results1, results2, results3, results4

def analyze_oncology():
    """Q005: Oncology immunotherapy."""
    print("\n" + "█"*80)
    print("  Q005: ONCOLOGY IMMUNOTHERAPY ANALYSIS")
    print("█"*80)

    # Count oncologists
    sql1 = """
    SELECT COUNT(*) as total_oncologists,
           MIN(cited_by_count) as min_cit,
           MAX(cited_by_count) as max_cit,
           ROUND(AVG(cited_by_count), 0) as avg_cit,
           MIN(h_index) as min_h,
           MAX(h_index) as max_h,
           ROUND(AVG(h_index), 1) as avg_h
    FROM authors
    WHERE concepts::text ILIKE '%oncolog%'
       OR concepts::text ILIKE '%cancer%';
    """

    results1 = run_query(sql1, "Oncology Researchers - Statistics")

    # Top oncologists
    sql2 = """
    SELECT display_name, h_index, cited_by_count, works_count,
           last_known_institution_name
    FROM authors
    WHERE concepts::text ILIKE '%oncolog%' OR concepts::text ILIKE '%cancer%'
    ORDER BY cited_by_count DESC
    LIMIT 10;
    """

    results2 = run_query(sql2, "Top 10 Oncology Researchers by Citations")

    # With immunotherapy focus
    sql3 = """
    SELECT COUNT(*) as count_with_immuno,
           MIN(cited_by_count) as min_cit,
           MAX(cited_by_count) as max_cit
    FROM authors
    WHERE (concepts::text ILIKE '%oncolog%' OR concepts::text ILIKE '%cancer%')
      AND concepts::text ILIKE '%immuno%';
    """

    results3 = run_query(sql3, "Oncologists with Immunotherapy Focus")

    # Check strict requirements
    sql4 = """
    SELECT COUNT(*) as count_strict
    FROM authors
    WHERE (concepts::text ILIKE '%oncolog%' OR concepts::text ILIKE '%cancer%')
      AND cited_by_count >= 500;
    """

    results4 = run_query(sql4, "Oncologists Meeting Strict Requirements (cit>=500)")

    return results1, results2, results3, results4

def analyze_quantum():
    """Q003, Q017: Quantum physics and computing."""
    print("\n" + "█"*80)
    print("  Q003 & Q017: QUANTUM PHYSICS & COMPUTING ANALYSIS")
    print("█"*80)

    # All quantum researchers
    sql1 = """
    SELECT COUNT(*) as total_quantum,
           MIN(h_index) as min_h,
           MAX(h_index) as max_h,
           ROUND(AVG(h_index), 1) as avg_h,
           MIN(cited_by_count) as min_cit,
           MAX(cited_by_count) as max_cit
    FROM authors
    WHERE concepts::text ILIKE '%quantum%';
    """

    results1 = run_query(sql1, "Quantum Researchers - Statistics")

    # Top quantum researchers
    sql2 = """
    SELECT display_name, h_index, cited_by_count, works_count,
           last_known_institution_name
    FROM authors
    WHERE concepts::text ILIKE '%quantum%'
    ORDER BY h_index DESC
    LIMIT 10;
    """

    results2 = run_query(sql2, "Top 10 Quantum Researchers by H-Index")

    # Quantum computing specifically
    sql3 = """
    SELECT COUNT(*) as count_quantum_computing,
           MAX(h_index) as max_h
    FROM authors
    WHERE concepts::text ILIKE '%quantum%'
      AND (concepts::text ILIKE '%comput%' OR concepts::text ILIKE '%algorithm%');
    """

    results3 = run_query(sql3, "Quantum Computing Researchers")

    # Check with any requirements
    sql4 = """
    SELECT display_name, h_index, cited_by_count,
           jsonb_array_elements(concepts)->>'display_name' as concept
    FROM authors
    WHERE concepts::text ILIKE '%quantum%'
      AND h_index >= 15
    ORDER BY h_index DESC
    LIMIT 10;
    """

    results4 = run_query(sql4, "Quantum Researchers with h>=15 (showing concepts)")

    return results1, results2, results3, results4

def analyze_robotics():
    """Q018: Robotics and industrial automation."""
    print("\n" + "█"*80)
    print("  Q018: ROBOTICS & AUTOMATION ANALYSIS")
    print("█"*80)

    sql1 = """
    SELECT COUNT(*) as total_robotics,
           MIN(h_index) as min_h,
           MAX(h_index) as max_h,
           ROUND(AVG(h_index), 1) as avg_h
    FROM authors
    WHERE concepts::text ILIKE '%robot%'
       OR concepts::text ILIKE '%automat%';
    """

    results1 = run_query(sql1, "Robotics/Automation Researchers - Statistics")

    sql2 = """
    SELECT display_name, h_index, cited_by_count, works_count,
           last_known_institution_name
    FROM authors
    WHERE concepts::text ILIKE '%robot%'
    ORDER BY cited_by_count DESC
    LIMIT 10;
    """

    results2 = run_query(sql2, "Top 10 Robotics Researchers by Citations")

    return results1, results2

def analyze_economics():
    """Q010: Behavioral economics."""
    print("\n" + "█"*80)
    print("  Q010: ECONOMICS (BEHAVIORAL) ANALYSIS")
    print("█"*80)

    sql1 = """
    SELECT COUNT(*) as total_economists,
           MIN(h_index) as min_h,
           MAX(h_index) as max_h,
           ROUND(AVG(h_index), 1) as avg_h
    FROM authors
    WHERE concepts::text ILIKE '%econom%';
    """

    results1 = run_query(sql1, "Economics Researchers - Statistics")

    sql2 = """
    SELECT display_name, h_index, cited_by_count, works_count,
           last_known_institution_name
    FROM authors
    WHERE concepts::text ILIKE '%econom%'
    ORDER BY h_index DESC
    LIMIT 10;
    """

    results2 = run_query(sql2, "Top 10 Economists by H-Index")

    sql3 = """
    SELECT COUNT(*) as count_behavioral,
           MAX(h_index) as max_h
    FROM authors
    WHERE concepts::text ILIKE '%econom%'
      AND (concepts::text ILIKE '%behavior%' OR concepts::text ILIKE '%decision%');
    """

    results3 = run_query(sql3, "Behavioral Economics Researchers")

    return results1, results2, results3

def main():
    """Run all analyses."""
    print("\n" + "█"*80)
    print("  🔬 SARA DATABASE COVERAGE ANALYSIS")
    print("  " + "─"*76)
    print(f"  Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("█"*80)

    results = {
        'metadata': {
            'timestamp': datetime.now().isoformat(),
            'database': 'devdb'
        },
        'analyses': {}
    }

    # Run all analyses
    print("\n\n🎯 PART 1: OVERALL DISTRIBUTIONS")
    h_dist, h_thresh = analyze_h_index_distribution()
    results['analyses']['h_index'] = {
        'distribution': [dict(r) for r in h_dist],
        'thresholds': dict(h_thresh[0]) if h_thresh else {}
    }

    cit_dist, cit_thresh = analyze_citation_distribution()
    results['analyses']['citations'] = {
        'distribution': [dict(r) for r in cit_dist],
        'thresholds': dict(cit_thresh[0]) if cit_thresh else {}
    }

    print("\n\n🎯 PART 2: DOMAIN-SPECIFIC ANALYSES")

    card_results = analyze_cardiology()
    results['analyses']['cardiology'] = {
        'stats': dict(card_results[0][0]) if card_results[0] else {},
        'top_10': [dict(r) for r in card_results[1]],
        'strict_count': dict(card_results[2][0]) if card_results[2] else {},
        'relaxed': [dict(r) for r in card_results[3]]
    }

    onco_results = analyze_oncology()
    results['analyses']['oncology'] = {
        'stats': dict(onco_results[0][0]) if onco_results[0] else {},
        'top_10': [dict(r) for r in onco_results[1]],
        'immuno_count': dict(onco_results[2][0]) if onco_results[2] else {},
        'strict_count': dict(onco_results[3][0]) if onco_results[3] else {}
    }

    quant_results = analyze_quantum()
    results['analyses']['quantum'] = {
        'stats': dict(quant_results[0][0]) if quant_results[0] else {},
        'top_10': [dict(r) for r in quant_results[1]],
        'computing_count': dict(quant_results[2][0]) if quant_results[2] else {}
    }

    robot_results = analyze_robotics()
    results['analyses']['robotics'] = {
        'stats': dict(robot_results[0][0]) if robot_results[0] else {},
        'top_10': [dict(r) for r in robot_results[1]]
    }

    econ_results = analyze_economics()
    results['analyses']['economics'] = {
        'stats': dict(econ_results[0][0]) if econ_results[0] else {},
        'top_10': [dict(r) for r in econ_results[1]],
        'behavioral_count': dict(econ_results[2][0]) if econ_results[2] else {}
    }

    # Save results
    output_file = f"database_coverage_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False, default=str)

    print("\n\n" + "█"*80)
    print("  ✅ ANALYSIS COMPLETE!")
    print("█"*80)
    print(f"\n💾 Full results saved to: {output_file}\n")

if __name__ == "__main__":
    main()
