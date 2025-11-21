"""Unit tests for the improved ActorSearchTool."""
import pytest

from cli.lib.agentic_rag import ActorSearchTool
from cli.lib.search_utils import load_movies


def test_search_respects_limit():
    """Test that search respects the limit parameter."""
    movies = load_movies()
    tool = ActorSearchTool(movies)

    results = tool.search("Actor Name", limit=3)

    assert len(results) <= 3


def test_search_returns_list():
    from cli.lib.inverted_index import InvertedIndex
    idx = InvertedIndex()
    movies = [
        {"id": 1, "title": "Paddington", "description": "Bear", "cast": ["Ben"], "genre": ["Comedy", "Family"]},
        {"id": 2, "title": "Ted", "description": "Bear", "cast": ["Mark"], "genre": ["Comedy"]},
    ]
    idx.movies = movies
    idx.build()
    results = idx.search("Bear", limit=2)
    assert isinstance(results, list)
    for m in results:
        assert isinstance(m["genre"], list)
        assert any(g in m["genre"] for g in ["Comedy", "Family"])
