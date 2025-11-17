"""Pattern-based search using regex."""

import re
from typing import Any

from .base import SearchTool


class RegexSearchTool(SearchTool):
    """Pattern-based search using regex."""

    def __init__(self, movies: list[dict[str, Any]]):
        super().__init__(
            name="regex_search",
            description="Finds movies matching text patterns like 'bear attack' or 'wilderness survival'. Best for finding specific phrases or word combinations."
        )
        self.movies = movies

    def search(self, query: str, limit: int = 10) -> list[dict[str, Any]]:
        # Convert query to a flexible regex pattern
        # Replace spaces with .* to allow words in any order with other words between
        pattern = re.compile(r'\b' + r'\b.*\b'.join(re.escape(word) for word in query.split()) + r'\b', re.IGNORECASE)

        matches = []
        for movie in self.movies:
            text = f"{movie['title']} {movie['description']}"
            if pattern.search(text):
                matches.append({
                    **movie,
                    'score': 1.0  # All matches have equal score for regex
                })

        return matches[:limit]

