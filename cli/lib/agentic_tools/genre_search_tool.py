"""Genre-based search with synonym and metadata support."""

from typing import Any

from .base import SearchTool
from .constants import GENRE_SYNONYMS
from ..search_utils import normalize_text


class GenreSearchTool(SearchTool):
    """Genre-based search with synonym and metadata support.

    Strategy:
    1. Parse the user query to detect one or more canonical genres
       using a synonym → canonical genre mapping.
    2. For each movie:
       - Prefer matching against an explicit `genres` field if present.
       - Fall back to matching synonyms in title/description text.
    3. Score movies higher when:
       - They match more of the requested genres
       - They match via explicit metadata instead of just text
    """

    def __init__(self, movies: list[dict[str, Any]]):
        super().__init__(
            name="genre_search",
            description=(
                "Filters movies by genres like horror, thriller, comedy, "
                "romance, sci-fi, etc. Understands synonyms such as 'suspense' "
                "→ thriller or 'science fiction' → sci-fi."
            ),
        )
        self.movies = movies

        # Use module-level GENRE_SYNONYMS
        self.genre_synonyms = GENRE_SYNONYMS

        # Build reverse map: normalized synonym → canonical genre
        self.synonym_to_canonical: dict[str, str] = {}
        for canon, syns in self.genre_synonyms.items():
            for syn in syns:
                self.synonym_to_canonical[self._normalize_text(syn)] = canon

    @staticmethod
    def _normalize_text(text: str) -> str:
        """Lowercase, strip punctuation, and collapse whitespace."""
        return normalize_text(text, strip_accents=False)

    def _extract_requested_genres(self, query: str) -> set[str]:
        """Return a set of canonical genres implied by the query."""
        q_norm = self._normalize_text(query)

        requested: set[str] = set()
        # Check each synonym; if it's contained in the normalized query,
        # map to its canonical genre.
        for syn_norm, canon in self.synonym_to_canonical.items():
            if syn_norm and syn_norm in q_norm:
                requested.add(canon)

        return requested

    def _get_movie_genres(self, movie: dict) -> list[str]:
        """Extract normalized canonical genres from movie metadata if present.

        Assumes movie may have:
        - movie["genres"]: list of strings (preferred)
        - or genre-like terms only in description/title (fallback handled separately)
        """
        raw_genres = movie.get("genres") or movie.get("genre") or []
        if isinstance(raw_genres, str):
            raw_genres = [raw_genres]

        normalized_genres: set[str] = set()
        for g in raw_genres:
            g_norm = self._normalize_text(str(g))
            # Try exact canonical match first
            if g_norm in self.genre_synonyms:
                normalized_genres.add(g_norm)
                continue

            # Try mapping via synonyms
            if g_norm in self.synonym_to_canonical:
                normalized_genres.add(self.synonym_to_canonical[g_norm])
                continue

            # As a fallback, see if any canonical genre is contained in the string
            for canon, syns in self.genre_synonyms.items():
                if canon in g_norm or any(self._normalize_text(s) in g_norm for s in syns):
                    normalized_genres.add(canon)

        return list(normalized_genres)

    def _score_movie_by_genre(
        self,
        requested_genres: set[str],
        movie: dict,
    ) -> tuple[float, list[str], str]:
        """Return (score, matched_genres, match_reason)."""
        if not requested_genres:
            return 0.0, [], "no_requested_genres"

        text = self._normalize_text(
            f"{movie.get('title', '')} {movie.get('description', '')}"
        )

        movie_genres = set(self._get_movie_genres(movie))

        # 1) Exact metadata-based matches (best signal)
        meta_matches = requested_genres & movie_genres

        # 2) Text-based matches via synonyms
        text_matches: set[str] = set()
        for canon in requested_genres:
            for syn in self.genre_synonyms.get(canon, []):
                syn_norm = self._normalize_text(syn)
                if syn_norm and syn_norm in text:
                    text_matches.add(canon)
                    break

        # Union for reporting
        matched_genres = sorted(meta_matches | text_matches)

        if not matched_genres:
            return 0.0, [], "no_match"

        # Scoring:
        # - Base score from fraction of requested genres matched
        # - Bonus if those matches come from explicit metadata
        # - Smaller bonus for text-only matches
        coverage = len(matched_genres) / max(1, len(requested_genres))

        # Start with coverage as base (0–1)
        score = coverage

        # Boost for metadata vs text
        if meta_matches:
            score += 0.4  # strong signal when metadata matches
        if text_matches:
            score += 0.2  # weaker signal for text-only matches

        # Clamp score to [0, 1]
        score = max(0.0, min(1.0, score))

        if meta_matches and text_matches:
            reason = "metadata_and_text"
        elif meta_matches:
            reason = "metadata_only"
        else:
            reason = "text_only"

        return score, matched_genres, reason

    def search(self, query: str, limit: int = 10) -> list[dict[str, Any]]:
        """Search movies by genre(s) inferred from the query."""
        requested_genres = self._extract_requested_genres(query)

        # If the user didn't clearly specify any known genre, bail out
        if not requested_genres:
            return []

        results: list[dict] = []

        for movie in self.movies:
            score, matched_genres, reason = self._score_movie_by_genre(
                requested_genres, movie
            )
            if score <= 0.0:
                continue

            result = movie.copy()
            result["score"] = score
            result["matched_genres"] = matched_genres
            result["genre_match_reason"] = reason
            results.append(result)

        # Sort by our genre score descending
        results.sort(key=lambda r: r.get("score", 0.0), reverse=True)

        return results[:limit]

