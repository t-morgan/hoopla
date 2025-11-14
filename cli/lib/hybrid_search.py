import json
import logging
import os
from typing import List, Any, Dict
import time

from sentence_transformers import CrossEncoder

from .inverted_index import InvertedIndex, INDEX_PATH
from .chunked_semantic_search import ChunkedSemanticSearch
from .llm_utils import execute_llm_prompt
from .search_utils import load_movies


logger = logging.getLogger(__name__)


class HybridSearch:
    def __init__(self, documents=None):
        if documents is None:
            documents = []
        self.documents = documents
        self.semantic_search = ChunkedSemanticSearch(max_chunk_size=8, overlap=2)
        self.semantic_search.load_or_create_chunk_embeddings(documents)

        self.idx = InvertedIndex()
        if not os.path.exists(INDEX_PATH):
            self.idx.build()
            self.idx.save()


    def _bm25_search(self, query, limit):
        self.idx.load()
        return self.idx.bm25_search(query, limit)


    def weighted_search(self, query, alpha, limit=5):
        bm25_results = self._bm25_search(query, limit*500)
        ss_results = self.semantic_search.search_chunks(query, limit=limit*500)
        # combine the results and normalize scores
        combined_results = {}
        bm25_scores = [res['score'] for res in bm25_results]
        ss_scores = [res['score'] for res in ss_results]
        norm_bm25_scores = normalize_vector(bm25_scores)
        norm_ss_scores = normalize_vector(ss_scores)
        for res, s in zip(bm25_results, norm_bm25_scores):
            combined_results[res['id']] = {
                'title': res['title'],
                'description': res['description'],
                'bm25_score': s,
                'ss_score': 0.0
            }
        for res, s in zip(ss_results, norm_ss_scores):
            if res['id'] in combined_results:
                combined_results[res['id']]['ss_score'] = s
            else:
                combined_results[res['id']] = {
                    'title': res['title'],
                    'description': res['description'],
                    'bm25_score': 0.0,
                    'ss_score': s
                }
        # add a hybrid score to each result
        for res in combined_results.values():
            res['hybrid_score'] = hybrid_score(res['bm25_score'], res['ss_score'], alpha)
        # sort by hybrid score
        sorted_results = sorted(combined_results.values(), key=lambda x: x['hybrid_score'], reverse=True)
        return sorted_results[:limit]


    def rrf_search(self, query, k, limit=10):
        bm25_results = self._bm25_search(query, limit*500)
        ss_results = self.semantic_search.search_chunks(query, limit=limit*500)
        combined_results = {}
        for rank, res in enumerate(bm25_results):
            score = rrf_score(rank + 1, k)
            combined_results[res['id']] = {
                'title': res['title'],
                'description': res['description'],
                'rrf_score': score,
                'bm25_rank': rank + 1,
                'ss_rank': None
            }
        for rank, res in enumerate(ss_results):
            score = rrf_score(rank + 1, k)
            if res['id'] in combined_results:
                combined_results[res['id']]['rrf_score'] += score
                combined_results[res['id']]['ss_rank'] = rank + 1
            else:
                combined_results[res['id']] = {
                    'title': res['title'],
                    'description': res['description'],
                    'rrf_score': score,
                    'bm25_rank': None,
                    'ss_rank': rank + 1
                }
        sorted_results = sorted(combined_results.values(), key=lambda x: x['rrf_score'], reverse=True)
        return sorted_results[:limit]


def normalize_vector(vector):
    min_score = min(vector) if vector else 0.0
    max_score = max(vector) if vector else 0.0
    return [(score - min_score) / (max_score - min_score) if max_score > min_score else 1.0 for score in vector]


def search_hybrid_weighted(query, alpha, limit=5):
    documents = load_movies()
    hs = HybridSearch(documents)
    results = hs.weighted_search(query, alpha, limit)
    for i, res in enumerate(results, 1):
        print(f"{i}. {res['title']} (Hybrid Score: {res['hybrid_score']:.4f})")
        print(f"Hybrid Score: {res['hybrid_score']:.4f}")
        print(f"BM25: {res['bm25_score']:.4f}, Semantic: {res['ss_score']:.4f}")
        print(f"{res['description']}\n")


def hybrid_score(bm25_score, semantic_score, alpha=0.5):
    return alpha * bm25_score + (1 - alpha) * semantic_score


def _parse_rerank_json(raw: str) -> dict:
    """
    Try to extract a JSON object/array from a messy LLM response.

    Handles things like:
    - Leading 'json' or 'JSON'
    - Explanatory text before/after
    - Stray code fences (if any slipped through)
    """
    if not raw:
        raise ValueError("Empty LLM response")

    cleaned = raw.strip()

    # If it starts with 'json' on the first line, strip it
    if cleaned.lower().startswith("json"):
        cleaned = cleaned[4:].lstrip()

    # Also handle stray backticks, just in case
    cleaned = cleaned.strip("`").strip()

    # Try direct json.loads first
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        # Last-resort: extract the first {...} or [...] block
        start_candidates = [cleaned.find("{"), cleaned.find("[")]
        start_candidates = [i for i in start_candidates if i != -1]
        if not start_candidates:
            raise

        start = min(start_candidates)
        end_brace = cleaned.rfind("}")
        end_bracket = cleaned.rfind("]")
        end_candidates = [i for i in (end_brace, end_bracket) if i != -1 and i >= start]
        if not end_candidates:
            raise

        end = max(end_candidates) + 1
        snippet = cleaned[start:end]

        return json.loads(snippet)


def _rerank_batched(
    query: str,
    results: List[Dict[str, Any]],
    batch_size: int = 10,
    max_batch_retries: int = 3,
    base_delay: float = 1.0,
) -> List[Dict[str, Any]]:
    """
    Batched LLM-based reranker with retry logic for each batch.
    """
    if not results:
        return results

    # Initialize rerank_score so we always have *some* value
    for doc in results:
        doc.setdefault("rerank_score", 0.0)

    for batch_start in range(0, len(results), batch_size):
        batch = results[batch_start: batch_start + batch_size]
        items_str_lines = []
        for i, doc in enumerate(batch):
            idx = i
            title = doc.get("title", "")
            description = doc.get("document") or doc.get("description", "")
            items_str_lines.append(
                f'{idx}. Title: "{title}"\n   Description: {description}'
            )
        items_str = "\n\n".join(items_str_lines)

        prompt = f"""\
You are scoring how well each movie matches the search query.

Query: "{query}"

Movies:
{items_str}

For each movie, rate how well it matches the query from 0 to 10
(10 = perfect match).

Respond ONLY with valid JSON in this exact format, with:
- NO code fences,
- NO leading 'json',
- NO additional text before or after the JSON.

{{
  "scores": [
    {{"index": 0, "score": 7.5}},
    {{"index": 1, "score": 9.0}}
  ]
}}
"""

        batch_success = False
        for attempt in range(max_batch_retries):
            raw = execute_llm_prompt(prompt)
            try:
                data = _parse_rerank_json(raw)
                scores = data.get("scores", [])
                batch_success = True
                break
            except Exception as e:  # noqa: BLE001
                logger.warning(
                    "Failed to parse batch rerank JSON (batch_start=%s, attempt=%d): %s; raw=%r",
                    batch_start,
                    attempt + 1,
                    e,
                    raw,
                )
                if attempt < max_batch_retries - 1:
                    sleep_for = base_delay * (2 ** attempt)
                    logger.warning(
                        "Retrying batch %d in %.1fs...", batch_start, sleep_for
                    )
                    time.sleep(sleep_for)
        if not batch_success:
            logger.error(
                "Batch rerank failed after %d attempts (batch_start=%d). Skipping batch.",
                max_batch_retries,
                batch_start,
            )
            continue

        # Apply scores back onto docs in this batch
        for score_item in scores:
            try:
                idx = int(score_item["index"])
                score_val = float(score_item["score"])
            except (KeyError, TypeError, ValueError):
                continue
            if 0 <= idx < len(batch):
                batch[idx]["rerank_score"] = score_val

    # Sort globally by rerank_score
    return sorted(results, key=lambda d: d["rerank_score"], reverse=True)


def _rerank_cross_encoder(query, results):
    model = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')

    texts = [f"{doc.get('title', '')} - {doc.get('document', '')}" for doc in results]
    queries = [query] * len(texts)
    scores = model.predict(list(zip(queries, texts)))

    for doc, score in zip(results, scores):
        doc['rerank_score'] = float(score)

    return sorted(results, key=lambda x: x['rerank_score'], reverse=True)


def _rerank_individual(query, results):
    for doc in results:
        prompt = f"""\
Rate how well this movie matches the search query.

Query: "{query}"
Movie: {doc.get("title", "")} - {doc.get("document", "")}

Consider:
- Direct relevance to query
- User intent (what they're looking for)
- Content appropriateness

Rate 0-10 (10 = perfect match).
Give me ONLY the number in your response, no other text or explanation.

Score:"""
        score = execute_llm_prompt(prompt)
        try:
            doc['rerank_score'] = float(score)
        except (TypeError, ValueError):
            doc['rerank_score'] = 0.0
    return sorted(results, key=lambda x: x['rerank_score'], reverse=True)


RERANK_MULTIPLIER = 5


def search_rrf(query, k=60, limit=10, rerank_method=None, debug=False):
    if debug:
        logging.basicConfig(level=logging.DEBUG)
        logger.debug(f"Original query: {query}")

    documents = load_movies()
    hs = HybridSearch(documents)

    base_limit = limit
    fetch_limit = base_limit

    if rerank_method is not None:
        fetch_limit *= RERANK_MULTIPLIER  # increase limit for reranking

    # Log query after enhancements (already enhanced in CLI)
    if debug:
        logger.debug(f"Query after enhancements: {query}")

    results = hs.rrf_search(query, k, fetch_limit)
    if debug:
        logger.debug(f"Results after RRF search: {json.dumps(results[:base_limit], indent=2)}")

    final_results = results
    if rerank_method is not None:
        if rerank_method == "batch":
            final_results = _rerank_batched(query, results)
        elif rerank_method == "cross_encoder":
            final_results = _rerank_cross_encoder(query, results)
        elif rerank_method == "individual":
            final_results = _rerank_individual(query, results)
        if debug:
            logger.debug(f"Final results after re-ranking: {json.dumps(final_results[:base_limit], indent=2)}")

    return final_results[:base_limit]


def print_results(results, show_rerank=False):
    for i, res in enumerate(results, 1):
        print(f"{i}. {res['title']}")
        if show_rerank:
            print(f"Rerank Score: {res.get('rerank_score', 'N/A')}")
        print(f"RRF Score: {res['rrf_score']:.4f}")
        print(f"BM25 Rank: {res['bm25_rank']}, Semantic Rank: {res['ss_rank']}")
        print(f"{res['description']}\n")

def rrf_score(rank, k=60):
    return 1 / (k + rank)
