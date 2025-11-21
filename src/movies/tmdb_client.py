import os
import requests
from typing import List, Dict, Optional

TMDB_BASE_URL = "https://api.themoviedb.org/3"

class TMDBApiError(Exception):
    def __init__(self, status_code: int, url: str, message: str = ""):
        super().__init__(f"TMDB API error {status_code} for URL {url}: {message}")
        self.status_code = status_code
        self.url = url
        self.message = message

def _get_api_key() -> str:
    api_key = os.getenv("TMDB_API_KEY")
    if not api_key:
        raise RuntimeError("TMDB_API_KEY environment variable is not set.")
    return api_key

def _tmdb_request(endpoint: str, params: Optional[dict] = None) -> dict:
    api_key = _get_api_key()
    url = f"{TMDB_BASE_URL}{endpoint}"
    if params is None:
        params = {}
    params["api_key"] = api_key
    response = requests.get(url, params=params)
    if response.status_code != 200:
        raise TMDBApiError(response.status_code, url, response.text)
    return response.json()

def search_movie_by_title(title: str, year: Optional[int] = None, language: str = "en-US") -> List[Dict]:
    """Call /search/movie and return the list of results."""
    params = {"query": title, "language": language}
    if year:
        params["year"] = year
    data = _tmdb_request("/search/movie", params)
    return data.get("results", [])

def get_movie_details(movie_id: int, language: str = "en-US") -> Dict:
    """Call /movie/{movie_id} with append_to_response=credits,external_ids to get details + cast + external IDs."""
    params = {"language": language, "append_to_response": "credits,external_ids"}
    return _tmdb_request(f"/movie/{movie_id}", params)

def get_popular_movies(page: int = 1, language: str = "en-US") -> Dict:
    """Call /movie/popular and return the response JSON."""
    params = {"page": page, "language": language}
    return _tmdb_request("/movie/popular", params)

def get_top_rated_movies(page: int = 1, language: str = "en-US") -> Dict:
    """Call /movie/top_rated and return the response JSON."""
    params = {"page": page, "language": language}
    return _tmdb_request("/movie/top_rated", params)
