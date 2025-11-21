import json
import os
from typing import Any
from movies.normalization import Movie
from .llm_utils import execute_llm_prompt


BM25_B = 0.75
BM25_K1 = 1.5
DEFAULT_SEARCH_LIMIT = 5

# Compute project root robustly
SCRIPT_PATH = os.path.abspath(__file__)
PROJECT_ROOT = SCRIPT_PATH
for _ in range(4):
    PROJECT_ROOT = os.path.dirname(PROJECT_ROOT)
DATA_PATH = os.path.join(PROJECT_ROOT, "data", "movies.json")
STOPWORDS_PATH = os.path.join(PROJECT_ROOT, "data", "stopwords.txt")
CACHE_DIR = os.path.join(PROJECT_ROOT, "cache")
INDEX_PATH = os.path.join(CACHE_DIR, "index.pkl")


def _validate_movie_record(movie: dict, idx: int) -> None:
    required_fields = {
        "id": int,
        "title": str,
        "description": str,
        "cast": list,
        "genre": list,
    }
    for field, expected_type in required_fields.items():
        if field not in movie:
            raise ValueError(f"Movie at index {idx} missing required field '{field}'")
        value = movie[field]
        if field == "cast":
            if not isinstance(value, list):
                raise TypeError(f"Movie at index {idx} field 'cast' must be a list")
            # Accept list of dicts with 'name' or list of strings
            if value and isinstance(value[0], dict):
                for c in value:
                    if not isinstance(c, dict) or "name" not in c:
                        raise TypeError(f"Movie at index {idx} field 'cast' must be a list of dicts with 'name' keys")
            elif value and not all(isinstance(c, str) for c in value):
                raise TypeError(f"Movie at index {idx} field 'cast' must be a list of strings or dicts with 'name'")
        elif field == "genre":
            # Accept legacy: if genre is a string, wrap in a list
            if isinstance(value, str):
                # Normalize legacy genre string to list
                movie["genre"] = [value]
                value = movie["genre"]
            if not isinstance(value, list) or not all(isinstance(g, str) for g in value):
                raise TypeError(f"Movie at index {idx} field 'genre' must be a list of strings")
        elif not isinstance(value, expected_type):
            raise TypeError(f"Movie at index {idx} field '{field}' must be of type {expected_type.__name__}")


def load_movies() -> list[Movie]:
    with open(DATA_PATH, "r") as f:
        data = json.load(f)
    if not isinstance(data, dict) or "movies" not in data or not isinstance(data["movies"], list):
        raise ValueError("data/movies.json must have shape { 'movies': [ ... ] }")
    movies = data["movies"]
    for idx, movie in enumerate(movies):
        _validate_movie_record(movie, idx)
    return movies


def load_stopwords() -> list[str]:
    with open(STOPWORDS_PATH, "r") as f:
        return f.read().splitlines()


def _expand_with_llm(query: str) -> str:
    """
    Expand a short search query using Google GenAI (Gemini).
    Falls back to the original query if the API is not configured or any error occurs.
    """
    prompt = f"""\
Expand this movie search query with related terms.

Add synonyms and related concepts that might appear in movie descriptions.
Keep expansions relevant and focused.
This will be appended to the original query.

Examples:
- "scary bear movie" -> "scary horror grizzly bear movie terrifying film"
- "action movie with bear" -> "action thriller bear chase fight adventure"
- "comedy with bear" -> "comedy funny bear humor lighthearted"

Query: "{query}"
"""
    return execute_llm_prompt(prompt) or query


def _rewrite_with_llm(query: str) -> str:
    """
    Rewrite a short search query using Google GenAI (Gemini).
    Falls back to the original query if the API is not configured or any error occurs.
    """
    prompt = f"""\
Rewrite the movie search query into a short, high-signal phrase optimized for hybrid search (BM25 + semantic search).

Original query: {query}

Rewrite Rules:
- Use only well-known facts or obvious inferences (actors, settings, genres, iconic elements).
- Focus on concrete keywords, not full sentences.
- Keep it under 10 words.
- Do not use quotes, asterisks, punctuation, or boolean operators.
- Avoid adding movie titles unless the query strongly implies it.
- Prefer descriptive keywords that strengthen both BM25 and embeddings:
    - actors
    - locations
    - objects
    - genre terms
    - era ranges (e.g., “2010s”)

Examples:
- that bear movie where leo gets attacked → revenant leonardo dicaprio bear attack
- movie about bear in london with marmalade → paddington london marmalade
- scary movie with bear from few years ago → bear horror 2015–2020
- cartoon dog solving mysteries → scooby doo cartoon mystery

Rewritten query:
"""
    return execute_llm_prompt(prompt) or query

def _spell_correct_with_llm(query: str) -> str:
    """
    Attempt to correct spelling in a short search query using Google GenAI (Gemini).
    Falls back to the original query if the API is not configured or any error occurs.
    """
    prompt = f"""\
You correct spelling mistakes in short movie search queries.

Rules:
- Correct misspellings, including proper nouns (movie titles, characters, actors, places).
- Prefer correcting to the nearest valid English spelling.
- Do not add or remove words.
- Return ONLY the corrected query text.
- If the query is already correct, return it unchanged.
- The domain is movies and entertainment.

Query: "{query}"
 """
    return execute_llm_prompt(prompt) or query


def enhance_query(query: str, method: str | None) -> str:
    """
    Enhance the query text according to the requested method.
    Currently supports:
      - "rewrite": LLM-powered query rewriting for better search effectiveness.
      - "spell": LLM-powered spelling correction.
    If method is None or unrecognized, returns the original query.
    """
    func = {
        "expand": _expand_with_llm,
        "rewrite": _rewrite_with_llm,
        "spell": _spell_correct_with_llm
    }.get(method)
    if func:
        enhanced_query = func(query)
        print(f"Enhanced query ({method}): '{query}' -> '{enhanced_query}'\n")
        return enhanced_query
    return query

def normalize_text(text: str, *, strip_accents: bool = False) -> str:
    """Normalize movie-related text: lowercase, strip punctuation, collapse whitespace, optionally strip accents."""
    import re, unicodedata
    if strip_accents:
        text = unicodedata.normalize("NFKD", text)
        text = "".join(c for c in text if not unicodedata.combining(c))
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text
