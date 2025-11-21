import os
import requests
import logging
import json
from typing import Optional
from functools import lru_cache

OMDB_API_URL = "http://www.omdbapi.com/"
OMDB_API_KEY = os.getenv("OMDB_API_KEY")

# Compute project root robustly
SCRIPT_PATH = os.path.abspath(__file__)
PROJECT_ROOT = SCRIPT_PATH
for _ in range(2):
    PROJECT_ROOT = os.path.dirname(PROJECT_ROOT)
CACHE_DIR = os.path.join(PROJECT_ROOT, "cache")
OMDB_CACHE_PATH = os.path.join(CACHE_DIR, ".omdb_cache.json")

OMDB_REQUEST_LIMIT = int(os.getenv("OMDB_REQUEST_LIMIT", "1000"))  # Default: 1000/day
OMDB_ONLY_MODE = bool(int(os.getenv("OMDB_ONLY_MODE", "0")))  # Set to 1 for omdb-only enrichment

if OMDB_API_KEY is None:
    raise RuntimeError("OMDB_API_KEY environment variable is required for OMDb enrichment.")

logger = logging.getLogger("omdb_client")

# --- Disk cache helpers ---
def _load_disk_cache() -> dict:
    try:
        with open(OMDB_CACHE_PATH, "r") as f:
            return json.load(f)
    except Exception:
        return {}

def _save_disk_cache(cache: dict) -> None:
    try:
        with open(OMDB_CACHE_PATH, "w") as f:
            json.dump(cache, f)
    except Exception as e:
        logger.warning(f"Failed to write OMDb disk cache: {e}")

_disk_cache = _load_disk_cache()
_omdb_requests_this_run = 0
_omdb_unauthorized = False

# --- In-memory LRU cache ---
@lru_cache(maxsize=256)
def _fetch_full_plot_by_imdb_id(imdb_id: str) -> Optional[str]:
    global _omdb_requests_this_run, _omdb_unauthorized
    if _omdb_unauthorized or _omdb_requests_this_run >= OMDB_REQUEST_LIMIT:
        logger.info("OMDb request limit reached or unauthorized; skipping request.")
        return None
    params = {
        "i": imdb_id,
        "plot": "full",
        "apikey": OMDB_API_KEY,
    }
    try:
        resp = requests.get(OMDB_API_URL, params=params, timeout=10)
        if resp.status_code == 401:
            _omdb_unauthorized = True
            logger.error("OMDb returned 401 Unauthorized; stopping further OMDb requests.")
            return None
        _omdb_requests_this_run += 1
        resp.raise_for_status()
        data = resp.json()
    except Exception as e:
        logger.warning(f"OMDb fetch by IMDb ID failed: {imdb_id} ({e})")
        return None
    if not isinstance(data, dict) or data.get("Response") == "False":
        logger.info(f"OMDb returned no result for IMDb ID: {imdb_id}")
        return None
    plot = data.get("Plot")
    if not plot or not isinstance(plot, str) or plot.strip().lower() in ("n/a", ""):
        logger.info(f"OMDb plot missing for IMDb ID: {imdb_id}")
        return None
    return plot.strip()

@lru_cache(maxsize=256)
def _fetch_full_plot_by_title(title: str, year: Optional[int] = None) -> Optional[str]:
    global _omdb_requests_this_run, _omdb_unauthorized
    if _omdb_unauthorized or _omdb_requests_this_run >= OMDB_REQUEST_LIMIT:
        logger.info("OMDb request limit reached or unauthorized; skipping request.")
        return None
    params = {
        "t": title,
        "plot": "full",
        "apikey": OMDB_API_KEY,
    }
    if year is not None:
        params["y"] = str(year)
    try:
        resp = requests.get(OMDB_API_URL, params=params, timeout=10)
        if resp.status_code == 401:
            _omdb_unauthorized = True
            logger.error("OMDb returned 401 Unauthorized; stopping further OMDb requests.")
            return None
        _omdb_requests_this_run += 1
        resp.raise_for_status()
        data = resp.json()
    except Exception as e:
        logger.warning(f"OMDb fetch by title failed: '{title}' ({e})")
        return None
    if not isinstance(data, dict) or data.get("Response") == "False":
        logger.info(f"OMDb returned no result for title: '{title}'")
        return None
    plot = data.get("Plot")
    if not plot or not isinstance(plot, str) or plot.strip().lower() in ("n/a", ""):
        logger.info(f"OMDb plot missing for title: '{title}'")
        return None
    return plot.strip()

# --- Public API with disk cache ---
def fetch_full_plot_by_imdb_id(imdb_id: str) -> Optional[str]:
    key = f"imdb:{imdb_id}"
    if key in _disk_cache:
        return _disk_cache[key]
    if OMDB_ONLY_MODE:
        # Only enrich missing cache, do not make new OMDb requests
        return None
    plot = _fetch_full_plot_by_imdb_id(imdb_id)
    if plot:
        _disk_cache[key] = plot
        _save_disk_cache(_disk_cache)
    return plot

def fetch_full_plot_by_title(title: str, year: Optional[int] = None) -> Optional[str]:
    key = f"title:{title.lower()}"
    if key in _disk_cache:
        return _disk_cache[key]
    if OMDB_ONLY_MODE:
        # Only enrich missing cache, do not make new OMDb requests
        return None
    plot = _fetch_full_plot_by_title(title, year)
    if plot:
        _disk_cache[key] = plot
        _save_disk_cache(_disk_cache)
    return plot

def omdb_requests_made() -> int:
    return _omdb_requests_this_run

def omdb_unauthorized() -> bool:
    return _omdb_unauthorized
