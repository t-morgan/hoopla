import os
import json
import tempfile
import pytest
from scripts.build_movies_json import build_movies_dataset

# --- Fixtures and helpers ---
FAKE_GOLDEN = {
    "test_cases": [
        {"query": "bear movie", "relevant_docs": ["Paddington", "Ted"]},
        {"query": "action", "relevant_docs": ["Die Hard"]}
    ]
}

FAKE_MOVIES = {
    "Paddington": {"id": 1, "title": "Paddington", "description": "Bear", "cast": ["Ben"], "genre": ["Comedy", "Family"]},
    "Ted": {"id": 2, "title": "Ted", "description": "Bear", "cast": ["Mark"], "genre": ["Comedy"]},
    "Die Hard": {"id": 3, "title": "Die Hard", "description": "Action", "cast": ["Bruce"], "genre": ["Action"]},
    "Extra": {"id": 4, "title": "Extra", "description": "Other", "cast": ["Someone"], "genre": ["Drama"]}
}

FAKE_POPULAR = [
    {"id": 4, "title": "Extra", "description": "Other", "cast": ["Someone"], "genre": ["Drama"]},
    {"id": 5, "title": "Another", "description": "Other", "cast": ["Another"], "genre": ["Drama"]}
]

# --- Monkeypatch targets ---
MODULE = "movies.tmdb_client"

def fake_search_movie_by_title(title, year=None, language="en-US"):
    # Return a fake TMDB search result
    t = title.strip().lower()
    for k, v in FAKE_MOVIES.items():
        if k.lower() == t:
            return [{"id": v["id"], "title": v["title"]}]
    return []

def fake_get_movie_details(movie_id, language="en-US"):
    for v in FAKE_MOVIES.values():
        if v["id"] == movie_id:
            return {
                "id": v["id"],
                "title": v["title"],
                "overview": v["description"],
                "credits": {"cast": [{"name": c} for c in v["cast"]]},
                "genres": [{"name": g} for g in v["genre"]]
            }
    if movie_id == 5:
        return {
            "id": 5,
            "title": "Another",
            "overview": "Other",
            "credits": {"cast": [{"name": "Another"}]},
            "genres": [{"name": "Drama"}]
        }
    raise Exception("Not found")

def fake_get_popular_movies(page=1, language="en-US"):
    return {"results": FAKE_POPULAR}

def fake_get_top_rated_movies(page=1, language="en-US"):
    return {"results": []}

# --- Tests ---
@pytest.fixture
def temp_golden_file(tmp_path):
    path = tmp_path / "golden_dataset.json"
    with open(path, "w") as f:
        json.dump(FAKE_GOLDEN, f)
    return str(path)

@pytest.fixture(autouse=True)
def patch_paths(monkeypatch, temp_golden_file):
    # Patch GOLDEN_PATH and OUTPUT_PATH in the pipeline
    monkeypatch.setattr("scripts.build_movies_json.GOLDEN_PATH", temp_golden_file)
    monkeypatch.setattr("scripts.build_movies_json.OUTPUT_PATH", temp_golden_file.replace("golden_dataset.json", "movies.json"))

# --- Main test: golden titles included ---
def test_golden_titles_included(monkeypatch):
    monkeypatch.setattr(f"{MODULE}.search_movie_by_title", fake_search_movie_by_title)
    monkeypatch.setattr(f"{MODULE}.get_movie_details", fake_get_movie_details)
    monkeypatch.setattr(f"{MODULE}.get_popular_movies", fake_get_popular_movies)
    monkeypatch.setattr(f"{MODULE}.get_top_rated_movies", fake_get_top_rated_movies)
    result = build_movies_dataset(limit=2, language="en-US")
    titles = {m["title"] for m in result["movies"]}
    for t in ["Paddington", "Ted", "Die Hard"]:
        assert t in titles
    assert len(result["movies"]) >= 3

# --- Test: missing golden title raises error ---
def test_missing_golden_title(monkeypatch):
    def search_missing(title, year=None, language="en-US"):
        if title.strip().lower() == "ted":
            return []
        return fake_search_movie_by_title(title, year, language)
    monkeypatch.setattr(f"{MODULE}.search_movie_by_title", search_missing)
    monkeypatch.setattr(f"{MODULE}.get_movie_details", fake_get_movie_details)
    monkeypatch.setattr(f"{MODULE}.get_popular_movies", fake_get_popular_movies)
    monkeypatch.setattr(f"{MODULE}.get_top_rated_movies", fake_get_top_rated_movies)
    with pytest.raises(RuntimeError) as e:
        build_movies_dataset(limit=2, language="en-US")
    assert "ted" in str(e.value).lower()

# --- Test: limit respected for non-golden movies ---
def test_limit_respected(monkeypatch):
    monkeypatch.setattr(f"{MODULE}.search_movie_by_title", fake_search_movie_by_title)
    monkeypatch.setattr(f"{MODULE}.get_movie_details", fake_get_movie_details)
    monkeypatch.setattr(f"{MODULE}.get_popular_movies", fake_get_popular_movies)
    monkeypatch.setattr(f"{MODULE}.get_top_rated_movies", fake_get_top_rated_movies)
    result = build_movies_dataset(limit=1, language="en-US")
    # Should have all goldens + at most 1 extra
    titles = {m["title"] for m in result["movies"]}
    for t in ["Paddington", "Ted", "Die Hard"]:
        assert t in titles
    assert len(result["movies"]) <= 4

# Test: genre list is joined in search text
from cli.lib.inverted_index import movie_to_search_text

def test_movie_to_search_text_genre_list():
    movie = {
        "id": 1,
        "title": "Paddington",
        "description": "Bear",
        "cast": ["Ben"],
        "genre": ["Comedy", "Family"]
    }
    text = movie_to_search_text(movie)
    assert "Comedy" in text and "Family" in text
    assert "," in text  # genres are joined with comma

# New test: multiple genres preserved from TMDB
from movies.normalization import normalize_movie_from_tmdb

def test_normalize_movie_from_tmdb_multiple_genres():
    details = {
        "id": 42,
        "title": "Test Movie",
        "overview": "A test movie.",
        "credits": {"cast": [{"name": "Actor One"}]},
        "genres": [
            {"id": 1, "name": "Comedy"},
            {"id": 2, "name": "Family"}
        ]
    }
    movie = normalize_movie_from_tmdb(details)
    assert movie["genre"] == ["Comedy", "Family"]

from movies.omdb_client import fetch_full_plot_by_imdb_id, fetch_full_plot_by_title

# --- OMDb enrichment tests ---
def test_omdb_full_plot_replaces_tmdb(monkeypatch):
    # OMDb returns a long plot, should override TMDB overview
    def fake_get_movie_details(movie_id, language="en-US"):
        return {
            "id": 1,
            "title": "Paddington",
            "overview": "Short TMDB text",
            "credits": {"cast": [{"name": "Ben Whishaw"}]},
            "genres": [{"name": "Comedy"}, {"name": "Family"}],
            "external_ids": {"imdb_id": "tt1234567"}
        }
    monkeypatch.setattr(f"{MODULE}.search_movie_by_title", lambda title, **_: [{"id": 1, "title": "Paddington"}])
    monkeypatch.setattr(f"{MODULE}.get_movie_details", fake_get_movie_details)
    monkeypatch.setattr(f"{MODULE}.get_popular_movies", fake_get_popular_movies)
    monkeypatch.setattr(f"{MODULE}.get_top_rated_movies", fake_get_top_rated_movies)
    monkeypatch.setattr("movies.omdb_client.fetch_full_plot_by_imdb_id", lambda imdb_id: "This is a long, detailed plot...")
    monkeypatch.setattr("movies.omdb_client.fetch_full_plot_by_title", lambda title: None)
    result = build_movies_dataset(limit=1, language="en-US")
    movie = next(m for m in result["movies"] if m["title"] == "Paddington")
    assert movie["description"] == "This is a long, detailed plot..."


def test_omdb_fallback_to_tmdb(monkeypatch):
    # OMDb returns None, should fallback to TMDB overview
    def fake_get_movie_details(movie_id, language="en-US"):
        return {
            "id": 1,
            "title": "Paddington",
            "overview": "Short TMDB text",
            "credits": {"cast": [{"name": "Ben Whishaw"}]},
            "genres": [{"name": "Comedy"}, {"name": "Family"}],
            "external_ids": {"imdb_id": "tt1234567"}
        }
    monkeypatch.setattr(f"{MODULE}.search_movie_by_title", lambda title, **_: [{"id": 1, "title": "Paddington"}])
    monkeypatch.setattr(f"{MODULE}.get_movie_details", fake_get_movie_details)
    monkeypatch.setattr(f"{MODULE}.get_popular_movies", fake_get_popular_movies)
    monkeypatch.setattr(f"{MODULE}.get_top_rated_movies", fake_get_top_rated_movies)
    monkeypatch.setattr("movies.omdb_client.fetch_full_plot_by_imdb_id", lambda imdb_id: None)
    monkeypatch.setattr("movies.omdb_client.fetch_full_plot_by_title", lambda title: None)
    result = build_movies_dataset(limit=1, language="en-US")
    movie = next(m for m in result["movies"] if m["title"] == "Paddington")
    assert movie["description"] == "Short TMDB text"


def test_omdb_imdb_id_path(monkeypatch):
    # OMDb should be queried by IMDb ID
    called = {}
    def fake_get_movie_details(movie_id, language="en-US"):
        return {
            "id": 1,
            "title": "Paddington",
            "overview": "Short TMDB text",
            "credits": {"cast": [{"name": "Ben Whishaw"}]},
            "genres": [{"name": "Comedy"}, {"name": "Family"}],
            "external_ids": {"imdb_id": "tt1234567"}
        }
    def fake_fetch_full_plot_by_imdb_id(imdb_id):
        called["imdb_id"] = imdb_id
        return "This is a long, detailed plot..."
    def fake_fetch_full_plot_by_title(title):
        called["title"] = title
        return None
    monkeypatch.setattr(f"{MODULE}.search_movie_by_title", lambda title, **_: [{"id": 1, "title": "Paddington"}])
    monkeypatch.setattr(f"{MODULE}.get_movie_details", fake_get_movie_details)
    monkeypatch.setattr(f"{MODULE}.get_popular_movies", fake_get_popular_movies)
    monkeypatch.setattr(f"{MODULE}.get_top_rated_movies", fake_get_top_rated_movies)
    monkeypatch.setattr("movies.omdb_client.fetch_full_plot_by_imdb_id", fake_fetch_full_plot_by_imdb_id)
    monkeypatch.setattr("movies.omdb_client.fetch_full_plot_by_title", fake_fetch_full_plot_by_title)
    result = build_movies_dataset(limit=1, language="en-US")
    assert called["imdb_id"] == "tt1234567"
    assert "title" not in called


def test_omdb_title_fallback(monkeypatch):
    # OMDb should be queried by title if imdb_id is missing
    called = {}
    def fake_get_movie_details(movie_id, language="en-US"):
        return {
            "id": 1,
            "title": "Paddington",
            "overview": "Short TMDB text",
            "credits": {"cast": [{"name": "Ben Whishaw"}]},
            "genres": [{"name": "Comedy"}, {"name": "Family"}],
            "external_ids": {"imdb_id": None}
        }
    def fake_fetch_full_plot_by_imdb_id(imdb_id):
        called["imdb_id"] = imdb_id
        return None
    def fake_fetch_full_plot_by_title(title):
        called["title"] = title
        return "This is a long, detailed plot..."
    monkeypatch.setattr(f"{MODULE}.search_movie_by_title", lambda title, **_: [{"id": 1, "title": "Paddington"}])
    monkeypatch.setattr(f"{MODULE}.get_movie_details", fake_get_movie_details)
    monkeypatch.setattr(f"{MODULE}.get_popular_movies", fake_get_popular_movies)
    monkeypatch.setattr(f"{MODULE}.get_top_rated_movies", fake_get_top_rated_movies)
    monkeypatch.setattr("movies.omdb_client.fetch_full_plot_by_imdb_id", fake_fetch_full_plot_by_imdb_id)
    monkeypatch.setattr("movies.omdb_client.fetch_full_plot_by_title", fake_fetch_full_plot_by_title)
    result = build_movies_dataset(limit=1, language="en-US")
    assert called["title"] == "Paddington"
    movie = next(m for m in result["movies"] if m["title"] == "Paddington")
    assert movie["description"] == "This is a long, detailed plot..."

# Golden rules are already tested above (titles included, missing raises)
