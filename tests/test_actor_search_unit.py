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
    """Test that search always returns a list."""
    movies = load_movies()
    tool = ActorSearchTool(movies)

    results = tool.search("Someone", limit=5)

    assert isinstance(results, list)
