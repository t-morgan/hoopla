import string
from nltk.stem import PorterStemmer

from .search_utils import (
    DEFAULT_SEARCH_LIMIT,
    load_movies,
    load_stopwords,
)


def search_command(query: str, limit: int = DEFAULT_SEARCH_LIMIT) -> list[dict]:
    movies = load_movies()
    results = []
    for movie in movies:
        query_tokens = tokenize_text(query)
        title_tokens = tokenize_text(movie["title"])
        if has_matching_token(query_tokens, title_tokens):
            results.append(movie)
            if len(results) >= limit:
                break
    return results


def has_matching_token(query_tokens: list[str], title_tokens: list[str]) -> bool:
    return any(q in t for q in query_tokens for t in title_tokens)


def preprocess_text(text: str) -> str:
    text = text.lower()
    translator = str.maketrans('', '', string.punctuation)
    text = text.translate(translator)
    return text


def tokenize_text(text: str) -> list[str]:
    text = preprocess_text(text)
    tokens = text.split()
    valid_tokens = []
    for token in tokens:
        if token:
            valid_tokens.append(token)
    stopwords = load_stopwords()
    filtered_words = []
    for word in valid_tokens:
        if word not in stopwords:
            filtered_words.append(word)
    stemmer = PorterStemmer()
    return [stemmer.stem(word) for word in filtered_words]