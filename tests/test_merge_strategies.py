"""Tests for AgenticRAG merge strategies."""
import pytest
from cli.lib.agentic_rag import AgenticRAG, SearchResult, AgenticSearchConfig


def test_merge_union_deduplicates():
    """Test that union merge deduplicates by movie ID."""
    rag = AgenticRAG()

    # Create mock search results with overlapping movies
    results1 = SearchResult(
        tool_name='semantic_search',
        query='test',
        results=[
            {'id': 1, 'title': 'Movie A', 'score': 0.9},
            {'id': 2, 'title': 'Movie B', 'score': 0.8},
        ]
    )
    results2 = SearchResult(
        tool_name='keyword_search',
        query='test',
        results=[
            {'id': 2, 'title': 'Movie B', 'score': 0.7},
            {'id': 3, 'title': 'Movie C', 'score': 0.6},
        ]
    )

    merged = rag._merge_union([results1, results2])

    # Should have 3 unique movies
    assert len(merged) == 3
    movie_ids = {m['id'] for m in merged}
    assert movie_ids == {1, 2, 3}


def test_merge_intersection_requires_multiple_matches():
    """Test that intersection only keeps movies found by multiple tools."""
    rag = AgenticRAG()

    # Create search results where only Movie B appears in both
    results1 = SearchResult(
        tool_name='actor_search',
        query='Tom Hanks',
        results=[
            {'id': 1, 'title': 'Movie A', 'score': 0.9},
            {'id': 2, 'title': 'Movie B', 'score': 0.85},
        ]
    )
    results2 = SearchResult(
        tool_name='genre_search',
        query='horror',
        results=[
            {'id': 2, 'title': 'Movie B', 'score': 0.8},
            {'id': 3, 'title': 'Movie C', 'score': 0.7},
        ]
    )

    merged = rag._merge_intersection([results1, results2])

    # Should only have Movie B (appears in both searches)
    assert len(merged) == 1
    assert merged[0]['id'] == 2
    assert merged[0]['title'] == 'Movie B'
    assert 'found_by' in merged[0]
    assert 'tool_scores' in merged[0]


def test_merge_intersection_averages_scores():
    """Test that intersection averages scores from different tools."""
    rag = AgenticRAG()

    results1 = SearchResult(
        tool_name='actor_search',
        query='Actor',
        results=[{'id': 1, 'title': 'Movie A', 'score': 0.9}]
    )
    results2 = SearchResult(
        tool_name='genre_search',
        query='Genre',
        results=[{'id': 1, 'title': 'Movie A', 'score': 0.7}]
    )

    merged = rag._merge_intersection([results1, results2])

    assert len(merged) == 1
    # Average of 0.9 and 0.7 is 0.8, plus small completeness bonus
    assert merged[0]['aggregate_score'] >= 0.8
    assert merged[0]['aggregate_score'] <= 0.91  # 0.8 + 0.1 bonus


def test_merge_auto_uses_intersection_for_actor_plus_genre():
    """Test that auto strategy uses intersection for actor + genre searches."""
    rag = AgenticRAG()

    results1 = SearchResult(
        tool_name='actor_search',
        query='Actor',
        results=[
            {'id': 1, 'title': 'Movie A', 'score': 0.9},
            {'id': 2, 'title': 'Movie B', 'score': 0.8},
        ]
    )
    results2 = SearchResult(
        tool_name='genre_search',
        query='Genre',
        results=[
            {'id': 2, 'title': 'Movie B', 'score': 0.7},
            {'id': 3, 'title': 'Movie C', 'score': 0.6},
        ]
    )

    merged = rag._merge_results([results1, results2], merge_strategy='auto')

    # Should use intersection - only Movie B
    assert len(merged) == 1
    assert merged[0]['id'] == 2


def test_merge_auto_uses_union_for_similar_tools():
    """Test that auto strategy uses union for similar search types."""
    rag = AgenticRAG()

    results1 = SearchResult(
        tool_name='semantic_search',
        query='test',
        results=[{'id': 1, 'title': 'Movie A', 'score': 0.9}]
    )
    results2 = SearchResult(
        tool_name='keyword_search',
        query='test',
        results=[{'id': 2, 'title': 'Movie B', 'score': 0.8}]
    )

    merged = rag._merge_results([results1, results2], merge_strategy='auto')

    # Should use union - both movies
    assert len(merged) == 2
    movie_ids = {m['id'] for m in merged}
    assert movie_ids == {1, 2}


def test_merge_single_search_returns_all():
    """Test that single search returns all its results."""
    rag = AgenticRAG()

    results = SearchResult(
        tool_name='actor_search',
        query='Actor',
        results=[
            {'id': 1, 'title': 'Movie A', 'score': 0.9},
            {'id': 2, 'title': 'Movie B', 'score': 0.8},
            {'id': 3, 'title': 'Movie C', 'score': 0.7},
        ]
    )

    merged = rag._merge_results([results], merge_strategy='auto')

    assert len(merged) == 3


def test_merge_intersection_three_tools_relaxed():
    """Test that with 3+ tools, intersection requires presence in at least 2."""
    rag = AgenticRAG()

    results1 = SearchResult(
        tool_name='actor_search',
        query='Actor',
        results=[
            {'id': 1, 'title': 'Movie A', 'score': 0.9},
            {'id': 2, 'title': 'Movie B', 'score': 0.85},
        ]
    )
    results2 = SearchResult(
        tool_name='genre_search',
        query='Genre',
        results=[
            {'id': 2, 'title': 'Movie B', 'score': 0.8},
            {'id': 3, 'title': 'Movie C', 'score': 0.75},
        ]
    )
    results3 = SearchResult(
        tool_name='keyword_search',
        query='Keyword',
        results=[
            {'id': 3, 'title': 'Movie C', 'score': 0.7},
            {'id': 4, 'title': 'Movie D', 'score': 0.6},
        ]
    )

    merged = rag._merge_intersection([results1, results2, results3])

    # Should have Movies B and C (each appears in 2 searches)
    assert len(merged) == 2
    movie_ids = {m['id'] for m in merged}
    assert movie_ids == {2, 3}


def test_merge_intersection_preserves_metadata():
    """Test that intersection merge preserves movie metadata."""
    rag = AgenticRAG()

    results1 = SearchResult(
        tool_name='actor_search',
        query='Actor',
        results=[
            {
                'id': 1,
                'title': 'Movie A',
                'description': 'A great movie',
                'score': 0.9,
                'matched_actors': [{'name': 'Actor Name'}]
            }
        ]
    )
    results2 = SearchResult(
        tool_name='genre_search',
        query='Genre',
        results=[
            {
                'id': 1,
                'title': 'Movie A',
                'description': 'A great movie',
                'score': 0.8
            }
        ]
    )

    merged = rag._merge_intersection([results1, results2])

    assert len(merged) == 1
    assert 'description' in merged[0]
    assert 'matched_actors' in merged[0]
    assert merged[0]['description'] == 'A great movie'

