import pytest

from cli.lib.agentic_rag import ActorSearchTool


@pytest.mark.slow
@pytest.mark.integration
def test_actor_search_full_name_match_returns_results(actor_search_tool):
    """Test that searching for a full actor name returns relevant results.

    Uses the shared actor_search_tool fixture to avoid re-importing heavy dependencies.
    """
    results = actor_search_tool.search("Jack Lemmon", limit=5)
    assert isinstance(results, list)
    assert len(results) > 0
    # Expect The China Syndrome among the results (contains Jack Lemmon in description)
    titles = [r["title"] for r in results]
    assert any("China Syndrome" in t for t in titles)


@pytest.mark.slow
@pytest.mark.integration
def test_actor_search_multiple_names_and_operator_and(actor_search_tool):
    """Test searching for multiple actors with AND semantics."""
    # Query with two co-stars from The China Syndrome
    results = actor_search_tool.search("Jane Fonda and Michael Douglas", limit=5)
    assert len(results) > 0
    titles = [r["title"] for r in results]
    assert any("China Syndrome" in t for t in titles)


@pytest.mark.slow
@pytest.mark.integration
def test_actor_search_single_token_penalized_but_possible(actor_search_tool):
    """Test that single-token searches still work without crashing."""
    # Last name only may still find, but we just ensure it doesn't crash and returns a list
    results = actor_search_tool.search("Lemmon", limit=5)
    assert isinstance(results, list)

