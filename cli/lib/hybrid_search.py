import os

from .inverted_index import InvertedIndex, INDEX_PATH
from .chunked_semantic_search import ChunkedSemanticSearch
from .llm_utils import execute_llm_prompt
from .search_utils import load_movies


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


def search_rrf(query, k=60, limit=10, rerank_method=None):
    documents = load_movies()
    hs = HybridSearch(documents)

    base_limit = limit
    fetch_limit = base_limit

    if rerank_method is not None:
        fetch_limit *= RERANK_MULTIPLIER  # increase limit for reranking

    results = hs.rrf_search(query, k, fetch_limit)

    if rerank_method is not None:
        if rerank_method == "individual":
            results = _rerank_individual(query, results)

    results = results[:base_limit]

    for i, res in enumerate(results, 1):
        print(f"{i}. {res['title']}")
        if rerank_method is not None:
            print(f"Rerank Score: {res.get('rerank_score', 'N/A')}")
        print(f"RRF Score: {res['rrf_score']:.4f}")
        print(f"BM25 Rank: {res['bm25_rank']}, Semantic Rank: {res['ss_rank']}")
        print(f"{res['description']}\n")


def rrf_score(rank, k=60):
    return 1 / (k + rank)
