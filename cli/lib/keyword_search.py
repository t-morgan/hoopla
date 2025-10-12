from .inverted_index import InvertedIndex
from .search_utils import (
    DEFAULT_SEARCH_LIMIT,
)
from .text_utils import tokenize_text


def build_command():
    index = InvertedIndex()
    index.build()
    index.save()


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
