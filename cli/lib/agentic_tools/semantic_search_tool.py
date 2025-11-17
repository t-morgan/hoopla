"""Semantic similarity search tool."""

from typing import Any

from .base import SearchTool
from ..semantic_search import search_movies as semantic_search


class SemanticSearchTool(SearchTool):
    """Semantic similarity search tool."""

    def __init__(self):
        super().__init__(
            name="semantic_search",
            description="Finds movies by semantic similarity using embeddings. Best for conceptual queries, themes, or when exact keywords might not match."
        )

    def search(self, query: str, limit: int = 10) -> list[dict[str, Any]]:
        return semantic_search(query, limit=limit)

