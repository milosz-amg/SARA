#!/usr/bin/env python3
"""
Run author validation queries through three retrieval strategies:
1. Qwen3-Embedding-0.6B author profile vectors (full profile embeddings).
2. SentenceTransformer story_text embeddings (intfloat/multilingual-e5-large).
3. Hybrid keyword filter + story_text semantic rerank.

The script reads queries from validation_queries.json and prints results.
Only author-type queries are processed; works queries are skipped.
"""

from __future__ import annotations

import argparse
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import psycopg2
import torch
from psycopg2.extras import RealDictCursor
from sentence_transformers import SentenceTransformer
from transformers import AutoModel, AutoTokenizer

# Local imports
from OpenAlex.scripts.llm_generator import DB_PARAMS  # type: ignore

# Constants
QWEN_EMBED_COLUMN = "embedding"
STORY_EMBED_COLUMN = "story_embedding"
KEYWORDS_EMBED_COLUMN = "keywords_embedding"
QUERY_FILE = Path("validation_queries.json")
STORY_MODEL_NAME = "intfloat/multilingual-e5-large"
QWEN_MODEL_NAME = "Qwen/Qwen3-Embedding-0.6B"
QWEN_MAX_LENGTH = 512
TOP_K = 5
DEFAULT_KEYWORD_WEIGHT = 0.3

logger = logging.getLogger(__name__)


@dataclass
class Query:
    id: str
    query_en: str
    query_pl: str
    search_type: str
    notes: Optional[str]
    expected_characteristics: Dict[str, Any]


def load_queries(path: Path) -> List[Query]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    queries = []
    for entry in data.get("queries", []):
        queries.append(
            Query(
                id=entry["id"],
                query_en=entry["query_en"],
                query_pl=entry.get("query_pl", ""),
                search_type=entry["search_type"],
                notes=entry.get("notes"),
                expected_characteristics=entry.get("expected_characteristics", {}),
            )
        )
    return queries


def connect_db():
    return psycopg2.connect(**DB_PARAMS)


def load_qwen_model():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(QWEN_MODEL_NAME, trust_remote_code=True)
    model = AutoModel.from_pretrained(QWEN_MODEL_NAME, trust_remote_code=True)
    model = model.to(device)
    model.eval()
    return tokenizer, model, device


def encode_qwen(tokenizer, model, device, text: str) -> np.ndarray:
    inputs = tokenizer(
        [text],
        padding=True,
        truncation=True,
        max_length=QWEN_MAX_LENGTH,
        return_tensors="pt",
    ).to(device)

    with torch.no_grad():
        outputs = model(**inputs)
        embeddings = outputs.last_hidden_state.mean(dim=1)
        embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)

    return embeddings[0].cpu().numpy()


def encode_story(model: SentenceTransformer, text: str) -> np.ndarray:
    return model.encode(text, normalize_embeddings=True)


def fetch_results(
    cursor,
    sql: str,
    params: Dict[str, Any],
) -> List[Dict[str, Any]]:
    cursor.execute(sql, params)
    return cursor.fetchall()


def search_qwen_embeddings(
    cursor,
    query_vec: np.ndarray,
    search_type: str,
    top_k: int,
) -> List[Dict[str, Any]]:
    if search_type != "authors":
        raise ValueError("Qwen embedding search currently supports only authors.")
    table = "authors"
    sql = f"""
        SELECT id,
               display_name,
               last_known_institution_name,
               concepts,
               {QWEN_EMBED_COLUMN} <=> %(query_vec)s::vector AS distance
        FROM {table}
        WHERE {QWEN_EMBED_COLUMN} IS NOT NULL
        ORDER BY distance
        LIMIT %(top_k)s
    """
    return fetch_results(
        cursor,
        sql,
        {"query_vec": query_vec.tolist(), "top_k": top_k},
    )


def search_story_embeddings(
    cursor,
    query_vec: np.ndarray,
    search_type: str,
    top_k: int,
) -> List[Dict[str, Any]]:
    if search_type != "authors":
        raise ValueError("Story embedding search currently supports only authors.")
    table = "authors"
    column = STORY_EMBED_COLUMN
    sql = f"""
        SELECT id,
               display_name,
               last_known_institution_name,
               concepts,
               {column} <=> %(query_vec)s::vector AS distance
        FROM {table}
        WHERE {column} IS NOT NULL
        ORDER BY distance
        LIMIT %(top_k)s
    """
    return fetch_results(cursor, sql, {"query_vec": query_vec.tolist(), "top_k": top_k})


def search_hybrid(
    cursor,
    story_vec: np.ndarray,
    keyword_vec: Optional[np.ndarray],
    search_type: str,
    keywords: Optional[Iterable[str]],
    top_k: int,
    keyword_weight: float,
) -> List[Dict[str, Any]]:
    if search_type != "authors":
        raise ValueError("Hybrid search currently supports only authors.")
    table = "authors"
    # Base SQL components
    where_clauses = [f"{STORY_EMBED_COLUMN} IS NOT NULL"]
    use_keyword_component = keyword_vec is not None and search_type == "authors"
    effective_kw_weight = keyword_weight if use_keyword_component else 0.0
    params: Dict[str, Any] = {
        "story_vec": story_vec.tolist(),
        "top_k": top_k,
        "kw_weight": effective_kw_weight,
    }

    score_sql = f"(1 - %(kw_weight)s) * ({STORY_EMBED_COLUMN} <=> %(story_vec)s::vector)"

    if use_keyword_component:
        where_clauses.append(f"{KEYWORDS_EMBED_COLUMN} IS NOT NULL")
        params["keyword_vec"] = keyword_vec.tolist()
        score_sql += " + %(kw_weight)s * ({KEYWORDS_EMBED_COLUMN} <=> %(keyword_vec)s::vector)".format(
            KEYWORDS_EMBED_COLUMN=KEYWORDS_EMBED_COLUMN
        )

    sql = f"""
        SELECT id,
               display_name,
               last_known_institution_name,
               concepts,
               {score_sql} AS distance
        FROM {table}
        WHERE {' AND '.join(where_clauses)}
        ORDER BY distance
        LIMIT %(top_k)s
    """
    return fetch_results(cursor, sql, params)


def get_keywords(expected: Dict[str, Any]) -> Optional[List[str]]:
    concepts = expected.get("concepts")
    if isinstance(concepts, list):
        return concepts
    return None


def split_name(full_name: Optional[str]) -> Tuple[Optional[str], Optional[str]]:
    if not full_name:
        return None, None
    parts = full_name.strip().split()
    if not parts:
        return None, None
    first = parts[0]
    last = parts[-1] if len(parts) > 1 else None
    return first, last


def extract_field_of_study(row: Dict[str, Any]) -> Optional[str]:
    concepts = row.get("concepts")
    if not concepts:
        return None
    try:
        if isinstance(concepts, str):
            concepts = json.loads(concepts)
        if isinstance(concepts, list):
            for concept in concepts:
                if isinstance(concept, dict):
                    name = concept.get("display_name")
                    if name:
                        return name
    except Exception:  # pragma: no cover - defensive
        return None
    return None


def format_result_row(row: Dict[str, Any]) -> str:
    full_name = row.get("display_name")
    first, last = split_name(full_name)
    institution = row.get("last_known_institution_name") or "-"
    field = extract_field_of_study(row) or "-"
    distance = row.get("distance")
    distance_str = f"{distance:.4f}" if isinstance(distance, (float, int)) else "N/A"
    return (
        f"Name: {first or '-'} | Surname: {last or '-'} | University: {institution} | "
        f"Field: {field} | Distance: {distance_str}"
    )


def run_for_query(
    cursor,
    story_model: SentenceTransformer,
    qwen_tokenizer,
    qwen_model,
    qwen_device,
    query: Query,
    top_k: int,
    keyword_weight: float,
) -> Dict[str, List[str]]:
    if query.search_type != "authors":
        raise ValueError(f"Unsupported search_type '{query.search_type}' for query {query.id}")
    story_vec = encode_story(story_model, query.query_en)
    qwen_vec = encode_qwen(qwen_tokenizer, qwen_model, qwen_device, query.query_en)
    keyword_vec = None
    keywords = get_keywords(query.expected_characteristics)
    if keywords and query.search_type == "authors":
        keyword_vec = encode_story(story_model, ", ".join(keywords))

    qwen_results = search_qwen_embeddings(cursor, qwen_vec, query.search_type, top_k)
    story_results = search_story_embeddings(cursor, story_vec, query.search_type, top_k)
    hybrid_results = search_hybrid(
        cursor,
        story_vec=story_vec,
        keyword_vec=keyword_vec,
        search_type=query.search_type,
        keywords=keywords,
        top_k=top_k,
        keyword_weight=keyword_weight,
    )

    return {
        "qwen": [format_result_row(r) for r in qwen_results],
        "story": [format_result_row(r) for r in story_results],
        "hybrid": [format_result_row(r) for r in hybrid_results],
    }


def main():
    parser = argparse.ArgumentParser(description="Compare search strategies using validation queries.")
    parser.add_argument("--top-k", type=int, default=TOP_K, help="How many results to display per strategy.")
    parser.add_argument(
        "--keyword-weight",
        type=float,
        default=DEFAULT_KEYWORD_WEIGHT,
        help="Weight for keywords embedding in hybrid search (0-1).",
    )
    parser.add_argument(
        "--query-id",
        type=str,
        help="Run only the given query id (e.g., Q001). By default run all queries.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        help="Run only the first N queries from the JSON.",
    )

    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    logger.info("Loading validation queries")
    queries = load_queries(QUERY_FILE)

    if args.query_id:
        queries = [q for q in queries if q.id == args.query_id]
    elif args.limit:
        queries = queries[: args.limit]

    if not queries:
        logger.error("No queries selected.")
        return

    logger.info("Loaded %d queries", len(queries))

    logger.info("Loading SentenceTransformer model %s", STORY_MODEL_NAME)
    story_model = SentenceTransformer(STORY_MODEL_NAME)
    logger.info("Loading Qwen embedding model %s", QWEN_MODEL_NAME)
    qwen_tokenizer, qwen_model, qwen_device = load_qwen_model()

    with connect_db() as conn:
        with conn.cursor(cursor_factory=RealDictCursor) as cursor:
            for query in queries:
                logger.info("Running query %s: %s", query.id, query.query_en)
                if query.search_type != "authors":
                    logger.info("Skipping query %s (search_type '%s' not supported)", query.id, query.search_type)
                    continue
                results = run_for_query(
                    cursor,
                    story_model=story_model,
                    qwen_tokenizer=qwen_tokenizer,
                    qwen_model=qwen_model,
                    qwen_device=qwen_device,
                    query=query,
                    top_k=args.top_k,
                    keyword_weight=args.keyword_weight,
                )
                print("=" * 80)
                print(f"Query {query.id} ({query.search_type})")
                print(f"PL: {query.query_pl}")
                print(f"EN: {query.query_en}")
                if query.notes:
                    print(f"Notes: {query.notes}")
                print("\nQwen profile embeddings:")
                print("\n".join(results["qwen"]) or "No results")
                print("\nStory_text embeddings:")
                print("\n".join(results["story"]) or "No results")
                print("\nHybrid (keywords + story_text):")
                print("\n".join(results["hybrid"]) or "No results")
                print("\n")


if __name__ == "__main__":
    main()
