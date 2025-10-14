from .inverted_index import InvertedIndex
from .search_utils import (
    DEFAULT_SEARCH_LIMIT,
)
from .text_utils import tokenize_text

def bm25_idf_command(term: str) -> float:
    index = InvertedIndex()
    index.load()

    return index.get_bm25_idf(term)


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

    query_tokens = tokenize_text(query)
    results = {}
    for token in query_tokens:
        matching_docs = index.get_documents(token)
        for doc in matching_docs:
            results[doc['id']] = doc
            if len(results) >= limit:
                break
        if len(results) >= limit:
            break
    return list(results.values())


def tf_command(doc_id: int, term: str) -> int:
    index = InvertedIndex()
    index.load()
    
    return index.get_tf(doc_id, term)


def tfidf_command(doc_id:int, term: str) -> float:
    index = InvertedIndex()
    index.load()

    return index.get_tf_idf(doc_id, term)
