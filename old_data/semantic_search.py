#!/usr/bin/env python3
"""
Semantic Search Module for SARA
Provides semantic search capabilities for authors and works using pgvector.
"""

import json
import logging
from typing import List, Dict, Optional, Tuple
import psycopg2
from psycopg2.extras import RealDictCursor
import torch
from transformers import AutoTokenizer, AutoModel

# Database connection parameters
DB_PARAMS = {
    'host': 'localhost',
    'port': 5432,
    'database': 'devdb',
    'user': 'devuser',
    'password': 'devpass'
}

# Model Configuration
MODEL_NAME = "Qwen/Qwen3-Embedding-0.6B"
EMBEDDING_DIMENSIONS = 1024

# Initialize global model variables
_tokenizer = None
_model = None
_device = None

def setup_logging():
    """Setup logging configuration."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

def load_embedding_model():
    """Load the Qwen embedding model (lazy loading)."""
    global _tokenizer, _model, _device

    if _model is not None:
        return _tokenizer, _model, _device

    logging.info(f"Loading embedding model: {MODEL_NAME}")

    _device = "cuda" if torch.cuda.is_available() else "cpu"
    logging.info(f"Using device: {_device}")

    _tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    _model = AutoModel.from_pretrained(MODEL_NAME, trust_remote_code=True)
    _model = _model.to(_device)
    _model.eval()

    logging.info("Model loaded successfully!")
    return _tokenizer, _model, _device

def get_query_embedding(query: str) -> List[float]:
    """
    Convert a text query into an embedding vector.

    Args:
        query: Text query to embed

    Returns:
        List of floats representing the embedding vector
    """
    tokenizer, model, device = load_embedding_model()

    with torch.no_grad():
        # Tokenize
        inputs = tokenizer(
            [query],
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt"
        ).to(device)

        # Get embeddings
        outputs = model(**inputs)

        # Use mean pooling on the last hidden state
        embeddings = outputs.last_hidden_state.mean(dim=1)

        # Normalize embeddings
        embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)

        return embeddings.cpu().tolist()[0]

def get_db_connection():
    """Get a database connection."""
    return psycopg2.connect(**DB_PARAMS, cursor_factory=RealDictCursor)

def search_similar_authors(
    query: str,
    top_k: int = 10,
    min_works: int = 0,
    min_citations: int = 0,
    institution: Optional[str] = None,
    boost_metrics: bool = True,
    candidate_pool_size: int = None
) -> List[Dict]:
    """
    Search for authors similar to the query using hybrid semantic + metric-based ranking.

    Args:
        query: Natural language description of what you're looking for
        top_k: Number of results to return
        min_works: Minimum number of works (filter)
        min_citations: Minimum number of citations (filter)
        institution: Filter by institution name (partial match)
        boost_metrics: If True, boost ranking by h-index and citations (default: True)
        candidate_pool_size: Size of candidate pool for reranking (default: top_k * 50)

    Returns:
        List of author dictionaries with similarity scores

    Example:
        >>> results = search_similar_authors(
        ...     "machine learning expert in computer vision",
        ...     top_k=5,
        ...     min_citations=1000
        ... )
    """
    logging.info(f"Searching for authors: '{query}' (boost_metrics={boost_metrics})")

    # Get query embedding
    query_embedding = get_query_embedding(query)

    # Determine candidate pool size for reranking
    if candidate_pool_size is None:
        # Default: fetch 50x more candidates if boosting metrics
        candidate_pool_size = top_k * 50 if boost_metrics else top_k

    # Build SQL query with filters
    filters = ["embedding IS NOT NULL"]
    params = [query_embedding]

    if min_works > 0:
        filters.append("works_count >= %s")
        params.append(min_works)

    if min_citations > 0:
        filters.append("cited_by_count >= %s")
        params.append(min_citations)

    if institution:
        filters.append("last_known_institution_name ILIKE %s")
        params.append(f"%{institution}%")

    where_clause = " AND ".join(filters)

    # Convert embedding to string format for pgvector
    embedding_str = str(query_embedding)

    # Build final params list to match SQL: [embedding_for_similarity, filter_params..., embedding_for_order, limit]
    # params[0] is query_embedding, so we skip it and take params[1:] which are the filter values
    filter_params = params[1:] if len(params) > 1 else []
    final_params = [embedding_str] + filter_params + [embedding_str, candidate_pool_size]

    sql = f"""
        SELECT
            id,
            openalex_id,
            orcid,
            display_name,
            works_count,
            cited_by_count,
            h_index,
            i10_index,
            last_known_institution_name,
            last_known_institution_country,
            concepts,
            1 - (embedding <=> %s::vector) as similarity
        FROM authors
        WHERE {where_clause}
        ORDER BY embedding <=> %s::vector
        LIMIT %s
    """

    conn = get_db_connection()
    cursor = conn.cursor()

    try:
        cursor.execute(sql, final_params)
        results = cursor.fetchall()

        # Convert to list of dicts and parse JSON fields
        candidates = []
        for row in results:
            author = dict(row)
            # Parse concepts if it's a string
            if author.get('concepts') and isinstance(author['concepts'], str):
                try:
                    author['concepts'] = json.loads(author['concepts'])
                except:
                    author['concepts'] = []
            candidates.append(author)

        # Apply hybrid ranking if boost_metrics is True
        if boost_metrics and len(candidates) > 0:
            # Calculate normalized scores
            max_h = max((a['h_index'] or 0) for a in candidates) or 1
            max_cit = max((a['cited_by_count'] or 0) for a in candidates) or 1

            for author in candidates:
                # Normalize metrics to [0, 1]
                h_norm = (author['h_index'] or 0) / max_h
                cit_norm = (author['cited_by_count'] or 0) / max_cit

                # Hybrid score: weighted combination
                # 60% semantic similarity, 25% h-index, 15% citations
                author['hybrid_score'] = (
                    0.60 * author['similarity'] +
                    0.25 * h_norm +
                    0.15 * cit_norm
                )

            # Rerank by hybrid score
            candidates.sort(key=lambda x: x['hybrid_score'], reverse=True)
            authors = candidates[:top_k]

            logging.info(f"Hybrid ranking: {len(candidates)} candidates → top {len(authors)}")
        else:
            # No boosting - just return top_k by similarity
            authors = candidates[:top_k]
            logging.info(f"Pure semantic search: {len(authors)} authors")

        return authors

    finally:
        cursor.close()
        conn.close()

def search_similar_works(
    query: str,
    top_k: int = 10,
    year_from: Optional[int] = None,
    year_to: Optional[int] = None,
    work_type: Optional[str] = None,
    min_citations: int = 0
) -> List[Dict]:
    """
    Search for works (publications) similar to the query using semantic search.

    Args:
        query: Natural language description of research topic/paper
        top_k: Number of results to return
        year_from: Minimum publication year
        year_to: Maximum publication year
        work_type: Filter by work type (e.g., 'article', 'book')
        min_citations: Minimum number of citations

    Returns:
        List of work dictionaries with similarity scores

    Example:
        >>> results = search_similar_works(
        ...     "deep learning for medical image segmentation",
        ...     top_k=10,
        ...     year_from=2020,
        ...     min_citations=10
        ... )
    """
    logging.info(f"Searching for works: '{query}'")

    # Get query embedding
    query_embedding = get_query_embedding(query)

    # Build SQL query with filters
    filters = ["embedding IS NOT NULL"]
    params = [query_embedding]

    if year_from:
        filters.append("publication_year >= %s")
        params.append(year_from)

    if year_to:
        filters.append("publication_year <= %s")
        params.append(year_to)

    if work_type:
        filters.append("type = %s")
        params.append(work_type)

    if min_citations > 0:
        filters.append("cited_by_count >= %s")
        params.append(min_citations)

    where_clause = " AND ".join(filters)

    # Convert embedding to string format for pgvector
    embedding_str = str(query_embedding)

    # Build final params list to match SQL: [embedding_for_similarity, filter_params..., embedding_for_order, top_k]
    filter_params = params[1:] if len(params) > 1 else []
    final_params = [embedding_str] + filter_params + [embedding_str, top_k]

    sql = f"""
        SELECT
            id,
            openalex_id,
            doi,
            title,
            display_name,
            publication_year,
            publication_date,
            type,
            cited_by_count,
            open_access_status,
            open_access_url,
            authorships,
            concepts,
            1 - (embedding <=> %s::vector) as similarity
        FROM works
        WHERE {where_clause}
        ORDER BY embedding <=> %s::vector
        LIMIT %s
    """

    conn = get_db_connection()
    cursor = conn.cursor()

    try:
        cursor.execute(sql, final_params)
        results = cursor.fetchall()

        # Convert to list of dicts and parse JSON fields
        works = []
        for row in results:
            work = dict(row)
            # Parse JSON fields if they're strings
            for field in ['authorships', 'concepts']:
                if work.get(field) and isinstance(work[field], str):
                    try:
                        work[field] = json.loads(work[field])
                    except:
                        work[field] = []
            works.append(work)

        logging.info(f"Found {len(works)} matching works")
        return works

    finally:
        cursor.close()
        conn.close()

def recommend_collaborators(
    project_description: str,
    top_k: int = 10,
    min_h_index: int = 0,
    institution: Optional[str] = None,
    exclude_institution: Optional[str] = None
) -> List[Dict]:
    """
    Recommend potential collaborators based on a project description.
    This is a specialized version of author search optimized for finding collaborators.

    Args:
        project_description: Natural language description of the research project
        top_k: Number of recommendations to return
        min_h_index: Minimum h-index (research impact threshold)
        institution: Prefer authors from this institution
        exclude_institution: Exclude authors from this institution

    Returns:
        List of author dictionaries with similarity scores and additional metrics

    Example:
        >>> results = recommend_collaborators(
        ...     "We need an expert in quantum computing and cryptography",
        ...     top_k=5,
        ...     min_h_index=20
        ... )
    """
    logging.info(f"Recommending collaborators for: '{project_description}'")

    # Enhance query for better collaborator matching
    enhanced_query = f"Research expertise: {project_description}"

    # Get query embedding
    query_embedding = get_query_embedding(enhanced_query)

    # Build SQL query with filters
    filters = ["embedding IS NOT NULL", "works_count > 5"]  # At least some publications
    params = [query_embedding]

    if min_h_index > 0:
        filters.append("h_index >= %s")
        params.append(min_h_index)

    if institution:
        filters.append("last_known_institution_name ILIKE %s")
        params.append(f"%{institution}%")

    if exclude_institution:
        filters.append("(last_known_institution_name NOT ILIKE %s OR last_known_institution_name IS NULL)")
        params.append(f"%{exclude_institution}%")

    where_clause = " AND ".join(filters)

    # Convert embedding to string format for pgvector
    embedding_str = str(query_embedding)

    # Build final params list to match SQL: [embedding_for_similarity, filter_params..., embedding_for_order, top_k]
    filter_params = params[1:] if len(params) > 1 else []
    final_params = [embedding_str] + filter_params + [embedding_str, top_k]

    sql = f"""
        SELECT
            id,
            openalex_id,
            orcid,
            display_name,
            works_count,
            cited_by_count,
            h_index,
            i10_index,
            last_known_institution_name,
            last_known_institution_country,
            concepts,
            1 - (embedding <=> %s::vector) as similarity,
            -- Add collaboration potential score
            CASE
                WHEN h_index >= 50 THEN 'High Impact'
                WHEN h_index >= 20 THEN 'Mid Impact'
                WHEN h_index >= 10 THEN 'Emerging'
                ELSE 'Early Career'
            END as impact_level
        FROM authors
        WHERE {where_clause}
        ORDER BY embedding <=> %s::vector
        LIMIT %s
    """

    conn = get_db_connection()
    cursor = conn.cursor()

    try:
        cursor.execute(sql, final_params)
        results = cursor.fetchall()

        # Convert to list of dicts and parse JSON fields
        collaborators = []
        for row in results:
            author = dict(row)
            # Parse concepts if it's a string
            if author.get('concepts') and isinstance(author['concepts'], str):
                try:
                    author['concepts'] = json.loads(author['concepts'])
                except:
                    author['concepts'] = []

            # Add top research areas for easy display
            if author.get('concepts'):
                author['top_research_areas'] = [
                    c.get('display_name', '')
                    for c in author['concepts'][:5]
                    if c.get('display_name')
                ]
            else:
                author['top_research_areas'] = []

            collaborators.append(author)

        logging.info(f"Found {len(collaborators)} potential collaborators")
        return collaborators

    finally:
        cursor.close()
        conn.close()

def get_author_details(author_id: str) -> Optional[Dict]:
    """
    Get detailed information about a specific author.

    Args:
        author_id: Author ID (internal or OpenAlex ID)

    Returns:
        Author dictionary or None if not found
    """
    conn = get_db_connection()
    cursor = conn.cursor()

    try:
        cursor.execute("""
            SELECT
                id, openalex_id, orcid, display_name,
                display_name_alternatives,
                works_count, cited_by_count, h_index, i10_index, oa_percent,
                last_known_institution_id, last_known_institution_name,
                last_known_institution_country,
                concepts, summary_stats, counts_by_year,
                created_date, updated_date
            FROM authors
            WHERE id = %s OR openalex_id = %s
        """, (author_id, author_id))

        row = cursor.fetchone()
        if not row:
            return None

        author = dict(row)

        # Parse JSON fields
        for field in ['display_name_alternatives', 'concepts', 'summary_stats', 'counts_by_year']:
            if author.get(field) and isinstance(author[field], str):
                try:
                    author[field] = json.loads(author[field])
                except:
                    author[field] = None

        return author

    finally:
        cursor.close()
        conn.close()

def get_work_details(work_id: str) -> Optional[Dict]:
    """
    Get detailed information about a specific work.

    Args:
        work_id: Work ID (internal or OpenAlex ID)

    Returns:
        Work dictionary or None if not found
    """
    conn = get_db_connection()
    cursor = conn.cursor()

    try:
        cursor.execute("""
            SELECT
                id, openalex_id, doi, title, display_name,
                publication_year, publication_date, language,
                type, type_crossref, cited_by_count,
                is_retracted, is_paratext,
                open_access_status, open_access_url,
                primary_location, best_oa_location,
                authorships, concepts, mesh, locations,
                abstract_inverted_index,
                grants, sustainable_development_goals,
                biblio, counts_by_year,
                created_date, updated_date
            FROM works
            WHERE id = %s OR openalex_id = %s
        """, (work_id, work_id))

        row = cursor.fetchone()
        if not row:
            return None

        work = dict(row)

        # Parse JSON fields
        json_fields = [
            'primary_location', 'best_oa_location', 'authorships',
            'concepts', 'mesh', 'locations', 'abstract_inverted_index',
            'grants', 'sustainable_development_goals', 'biblio', 'counts_by_year'
        ]
        for field in json_fields:
            if work.get(field) and isinstance(work[field], str):
                try:
                    work[field] = json.loads(work[field])
                except:
                    work[field] = None

        return work

    finally:
        cursor.close()
        conn.close()

if __name__ == "__main__":
    # Simple CLI for testing
    import sys

    setup_logging()

    if len(sys.argv) < 2:
        print("Usage:")
        print("  python semantic_search.py 'your search query'")
        print("\nExample:")
        print("  python semantic_search.py 'machine learning expert'")
        sys.exit(1)

    query = sys.argv[1]

    print("\n" + "="*80)
    print("SEMANTIC SEARCH DEMO")
    print("="*80)

    # Search authors
    print(f"\n🔍 Searching for authors matching: '{query}'")
    authors = search_similar_authors(query, top_k=5)

    print(f"\n📊 Top {len(authors)} matching authors:\n")
    for i, author in enumerate(authors, 1):
        print(f"{i}. {author['display_name']}")
        print(f"   Institution: {author.get('last_known_institution_name', 'N/A')}")
        print(f"   Metrics: {author['works_count']} works, {author['cited_by_count']} citations, h-index: {author['h_index']}")
        print(f"   Similarity: {author['similarity']:.4f}")

        # Show top research areas
        if author.get('concepts'):
            areas = [c.get('display_name', '') for c in author['concepts'][:3] if c.get('display_name')]
            if areas:
                print(f"   Research: {', '.join(areas)}")
        print()
