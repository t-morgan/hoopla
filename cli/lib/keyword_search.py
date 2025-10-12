# keyword_search.py
from .inverted_index import InvertedIndex
from .search_utils import (
    DEFAULT_SEARCH_LIMIT,
    load_movies,  # used by the simple scan search
)
from .text_utils import tokenize_text, has_matching_token


def build_command():
    index = InvertedIndex()
    index.build()
    index.save()
    print(f"First document for token 'merida' = {index.get_documents('merida')[0]}")


def search_command(query: str, limit: int = DEFAULT_SEARCH_LIMIT) -> list[dict]:
    movies = load_movies()
    query_tokens = tokenize_text(query)
    results = []
    for movie in movies:
        title_tokens = tokenize_text(movie["title"])
        if has_matching_token(query_tokens, title_tokens):
            results.append(movie)
            if len(results) >= limit:
                break
    return results
