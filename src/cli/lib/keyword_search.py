from .inverted_index import InvertedIndex
from .search_utils import (
    BM25_B,
    BM25_K1,
    DEFAULT_SEARCH_LIMIT,
)

def bm25_idf_command(term: str) -> float:
    index = InvertedIndex()
    index.load()

    return index.get_bm25_idf(term)

def bm25_search_command(query: str, limit: int = DEFAULT_SEARCH_LIMIT) -> list[dict]:
    index = InvertedIndex()
    index.load()

    return index.bm25_search(query, limit)


def bm25_tf_command(doc_id: int, term: str, k1: float = BM25_K1, b: float = BM25_B) -> float:
    index = InvertedIndex()
    index.load()

    return index.get_bm25_tf(doc_id, term, k1, b)


def build_command() -> None:
    index = InvertedIndex()
    index.build()
    index.save()


def idf_command(term: str) -> float:
    index = InvertedIndex()
    index.load()
    
    return index.get_idf(term)


def search_command(query: str, limit: int = DEFAULT_SEARCH_LIMIT) -> list[dict]:
    index = InvertedIndex()
    index.load()

    return index.search(query, limit)


def tf_command(doc_id: int, term: str) -> int:
    index = InvertedIndex()
    index.load()
    
    return index.get_tf(doc_id, term)


def tfidf_command(doc_id:int, term: str) -> float:
    index = InvertedIndex()
    index.load()

    return index.get_tf_idf(doc_id, term)
