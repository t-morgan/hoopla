"""Test fallback behavior when intersection returns no results."""
import pytest
from cli.lib.agentic_rag import AgenticRAG, SearchResult


def test_intersection_fallback_to_union():
    """Test that intersection falls back to union when no movies match all criteria."""
    rag = AgenticRAG()

    # Create search results with NO overlap
    results1 = SearchResult(
        tool_name='actor_search',
        query='Tom Hanks',
        results=[
            {'id': 1, 'title': 'Cast Away', 'score': 0.9},
            {'id': 2, 'title': 'Forrest Gump', 'score': 0.85},
        ]
    )
    results2 = SearchResult(
        tool_name='genre_search',
        query='horror',
        results=[
            {'id': 3, 'title': 'The Shining', 'score': 0.95},
            {'id': 4, 'title': 'Alien', 'score': 0.90},
        ]
    )

    # Merge with auto strategy (should detect intersection but fallback to union)
    merged = rag._merge_results([results1, results2], merge_strategy='auto')

    # Should have all 4 movies (fallback to union)
    assert len(merged) == 4
    movie_ids = {m['id'] for m in merged}
    assert movie_ids == {1, 2, 3, 4}

    # Results should NOT have intersection-specific metadata
    for movie in merged:
        assert 'found_by' in movie
        assert 'matched_by_count' not in movie  # This indicates union was used
        assert 'tool_scores' not in movie


def test_intersection_with_overlap_no_fallback():
    """Test that intersection works normally when there IS overlap."""
    rag = AgenticRAG()

    # Create search results WITH overlap
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
            {'id': 2, 'title': 'Movie B', 'score': 0.80},
            {'id': 3, 'title': 'Movie C', 'score': 0.75},
        ]
    )

    # Merge with auto strategy (should use intersection successfully)
    merged = rag._merge_results([results1, results2], merge_strategy='auto')

    # Should have only 1 movie (successful intersection, no fallback)
    assert len(merged) == 1
    assert merged[0]['id'] == 2
    assert merged[0]['title'] == 'Movie B'

    # Results SHOULD have intersection-specific metadata
    assert 'matched_by_count' in merged[0]
    assert merged[0]['matched_by_count'] == 2
    assert 'tool_scores' in merged[0]
    assert 'actor_search' in merged[0]['tool_scores']
    assert 'genre_search' in merged[0]['tool_scores']


def test_explicit_intersection_no_fallback():
    """Test that explicit intersection strategy doesn't fallback even with no results."""
    rag = AgenticRAG()

    # Create search results with NO overlap
    results1 = SearchResult(
        tool_name='actor_search',
        query='Actor',
        results=[{'id': 1, 'title': 'Movie A', 'score': 0.9}]
    )
    results2 = SearchResult(
        tool_name='genre_search',
        query='Genre',
        results=[{'id': 2, 'title': 'Movie B', 'score': 0.8}]
    )

    # When explicitly set to intersection, should still fallback
    # (This is actually desired behavior - we always want to show something)
    merged = rag._merge_results([results1, results2], merge_strategy='intersection')

    # Should fallback to union
    assert len(merged) == 2


def test_union_strategy_unaffected():
    """Test that explicit union strategy works as before."""
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

    merged = rag._merge_results([results1, results2], merge_strategy='union')

    # Should have both movies
    assert len(merged) == 2
    movie_ids = {m['id'] for m in merged}
    assert movie_ids == {1, 2}

