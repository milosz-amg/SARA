"""
ArXiv Category Hierarchy Utilities

Provides distance/similarity functions based on the ArXiv category taxonomy.
Three-level hierarchy:
  - Main group (8): cs, econ, eess, math, physics, q-bio, q-fin, stat
  - Archive: hep-ph, astro-ph, cs, math, nlin, cond-mat, ...
  - Subcategory: cs.AI, math.AG, astro-ph.CO, ...

Distance levels:
  0.0 = same subcategory
  0.33 = same archive, different subcategory
  0.67 = same main group, different archive
  1.0 = different main group
"""

from typing import List

# All archives that belong to the Physics main group
PHYSICS_ARCHIVES = [
    'astro-ph', 'cond-mat', 'gr-qc', 'hep-ex', 'hep-lat', 'hep-ph',
    'hep-th', 'math-ph', 'nlin', 'nucl-ex', 'nucl-th', 'physics',
    'quant-ph'
]

# Distance weights (tunable)
DIST_SAME_SUB = 0.0
DIST_SAME_ARCHIVE = 0.33
DIST_SAME_GROUP = 0.67
DIST_DIFF_GROUP = 1.0


def extract_main_category(subcategory: str) -> str:
    """
    Map subcategory to one of 8 main ArXiv groups.

    Groups: cs, econ, eess, math, physics, q-bio, q-fin, stat
    """
    prefix = subcategory.split('.')[0] if '.' in subcategory else subcategory
    if prefix in PHYSICS_ARCHIVES:
        return 'physics'
    return prefix


def extract_archive(subcategory: str) -> str:
    """
    Extract the archive level from a subcategory.

    Examples:
        cs.AI -> cs
        hep-ph -> hep-ph  (no dot = archive is the full string)
        astro-ph.CO -> astro-ph
        cond-mat.stat-mech -> cond-mat
    """
    if '.' in subcategory:
        return subcategory.split('.')[0]
    return subcategory


def category_distance(cat_a: str, cat_b: str) -> float:
    """
    Compute distance between two ArXiv subcategories.

    Returns:
        0.0  - same subcategory
        0.33 - same archive, different subcategory
        0.67 - same main group, different archive
        1.0  - different main group
    """
    if cat_a == cat_b:
        return DIST_SAME_SUB

    archive_a = extract_archive(cat_a)
    archive_b = extract_archive(cat_b)
    if archive_a == archive_b:
        return DIST_SAME_ARCHIVE

    group_a = extract_main_category(cat_a)
    group_b = extract_main_category(cat_b)
    if group_a == group_b:
        return DIST_SAME_GROUP

    return DIST_DIFF_GROUP


def category_similarity(cat_a: str, cat_b: str) -> float:
    """Similarity between two categories. 1.0 - distance."""
    return 1.0 - category_distance(cat_a, cat_b)


def multilabel_similarity(cats_a: List[str], cats_b: List[str]) -> float:
    """
    Compute fuzzy similarity between two papers using all their categories.

    Combines:
      - Jaccard overlap of category sets
      - Average best hierarchical match (symmetric)

    Returns float in [0.0, 1.0].
    """
    if not cats_a or not cats_b:
        return 0.0

    set_a = set(cats_a)
    set_b = set(cats_b)

    # Jaccard overlap
    intersection = len(set_a & set_b)
    union = len(set_a | set_b)
    jaccard = intersection / union if union > 0 else 0.0

    # Average best hierarchical match (symmetric)
    # For each cat in A, find best match in B
    best_a = sum(max(category_similarity(a, b) for b in cats_b) for a in cats_a) / len(cats_a)
    # For each cat in B, find best match in A
    best_b = sum(max(category_similarity(b, a) for a in cats_a) for b in cats_b) / len(cats_b)
    avg_best = (best_a + best_b) / 2.0

    return 0.5 * jaccard + 0.5 * avg_best
