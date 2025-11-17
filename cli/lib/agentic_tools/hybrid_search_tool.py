"""Hybrid RRF search combining keyword and semantic."""

from typing import Any

from .base import SearchTool
from ..hybrid_search import search_rrf


class HybridSearchTool(SearchTool):
    """Hybrid RRF search combining keyword and semantic."""

    def __init__(self, movies: list[dict[str, Any]]):
        super().__init__(
            name="hybrid_search",
            description="Combines keyword and semantic search using Reciprocal Rank Fusion. Best for balanced queries that benefit from both approaches."
        )
        self.movies = movies

    def search(self, query: str, limit: int = 10) -> list[dict[str, Any]]:
        return search_rrf(query, limit=limit, movies=self.movies)

