"""Actor-based search built on top of BM25 keyword search."""

import re
from difflib import SequenceMatcher
from typing import Any

from .base import SearchTool
from ..keyword_search import bm25_search_command
from ..search_utils import normalize_text


class ActorSearchTool(SearchTool):
    """Actor-based search built on top of BM25 keyword search.

    Strategy:
    - Parse potential actor names from the query.
    - Use BM25 to retrieve candidate movies for those names.
    - Re-rank/filter candidates based on how strongly the actor's name
      appears in the title/description (full name, last name, etc.).
    """

    def __init__(self, movies: list[dict[str, Any]]):
        super().__init__(
            name="actor_search",
            description=(
                "Finds movies mentioning specific actors in the title or "
                "description, using BM25 under the hood."
            ),
        )
        self.movies = movies

    @staticmethod
    def _normalize_text(text: str) -> str:
        """Lowercase, strip punctuation, and collapse whitespace."""
        return normalize_text(text, strip_accents=True)

    def _parse_actor_names(self, query: str) -> list[str]:
        """Very simple actor name extractor."""
        query = query.strip()
        if not query:
            return []
        # Strip filler
        cleaned = re.sub(
            r"\b(movies?|films?|with|starring|featuring|actors?|about|by|in)\b",
            " ",
            query,
            flags=re.IGNORECASE,
        )
        separator = re.compile(r"\s*(?:,|\band\b|\bor\b|&)\s*", re.IGNORECASE)
        parts = separator.split(cleaned)

        names: list[str] = []
        for p in parts:
            norm = self._normalize_text(p)
            if not norm:
                continue
            # require either 2+ tokens or a reasonably long single token
            if len(norm.split()) >= 2 or len(norm) >= 4:
                names.append(norm)
        return names

    def _actor_strength(self, actor_name_norm: str, movie: dict) -> float:
        text_raw = f"{movie.get('title', '')} {movie.get('description', '')}"
        text_norm = self._normalize_text(text_raw)
        # Full-name substring is strongest
        if actor_name_norm in text_norm:
            return 1.0
        # Last-name heuristic
        tokens = actor_name_norm.split()
        if isinstance(tokens, list) and len(tokens) >= 2:
            last = tokens[-1]
            if last in text_norm:
                return 0.9
        # Fuzzy fallback (global; crude but fine as a backup)
        ratio = SequenceMatcher(None, actor_name_norm, text_norm).ratio()
        if ratio >= 0.8:
            return ratio
        return 0.0

    def search(self, query: str, limit: int = 10) -> list[dict[str, Any]]:
        actor_names = self._parse_actor_names(query)
        if not actor_names:
            return []

        # Use the full actor phrase as the BM25 query for recall
        bm25_query = " ".join(actor_names)
        # Pull more than we need so we can filter
        candidates = bm25_search_command(bm25_query, limit=limit * 3) or []

        scored_results: list[dict] = []
        for movie in candidates:
            # Combine strength across all actor names (OR semantics)
            strengths = [
                self._actor_strength(name, movie)
                for name in actor_names
            ]
            if not strengths or max(strengths) < 0.75:
                continue  # too weak to be a real actor mention

            actor_score = max(strengths)
            base_score = float(movie.get("score", 1.0))  # BM25 score

            # Blend BM25 score and actor match strength
            # (tweak weights as you like)
            aggregate = 0.6 * actor_score + 0.4 * (base_score / (base_score + 1.0))

            movie_copy = movie.copy()
            movie_copy["score"] = aggregate
            movie_copy["actor_match_strength"] = actor_score
            movie_copy["actor_query_names"] = actor_names
            scored_results.append(movie_copy)

        scored_results.sort(key=lambda m: m.get("score", 0.0), reverse=True)
        return scored_results[:limit]

