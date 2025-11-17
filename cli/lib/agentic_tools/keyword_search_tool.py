"""BM25 keyword-based search tool."""

from typing import Any

from .base import SearchTool
from ..keyword_search import bm25_search_command


class KeywordSearchTool(SearchTool):
    """BM25 keyword-based search tool."""

    def __init__(self):
        super().__init__(
            name="keyword_search",
            description="Finds movies by exact keyword matching using BM25. Best for specific terms, titles, or phrases that appear in descriptions."
        )

    def search(self, query: str, limit: int = 10) -> list[dict[str, Any]]:
        return bm25_search_command(query, limit=limit)

