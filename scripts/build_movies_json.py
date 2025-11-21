import dotenv
dotenv.load_dotenv()

import argparse
import json
import os
from typing import Dict, List, Set

from movies.tmdb_client import (
    search_movie_by_title,
    get_movie_details,
    get_popular_movies,
    get_top_rated_movies,
)
from movies.normalization import normalize_movie_from_tmdb, Movie
from movies.omdb_client import fetch_full_plot_by_imdb_id, fetch_full_plot_by_title

# Robustly compute project root (repo root)
SCRIPT_PATH = os.path.abspath(__file__)
PROJECT_ROOT = SCRIPT_PATH
for _ in range(2):
    PROJECT_ROOT = os.path.dirname(PROJECT_ROOT)
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
GOLDEN_PATH = os.path.join(DATA_DIR, "golden_dataset.json")
OUTPUT_PATH = os.path.join(DATA_DIR, "movies.json")


def build_movies_dataset(limit: int, language: str) -> Dict:
    # Step 1: Load golden titles
    with open(GOLDEN_PATH, "r") as f:
        golden_data = json.load(f)
    golden_titles = set()
    for tc in golden_data.get("test_cases", []):
        for title in tc.get("relevant_docs", []):
            golden_titles.add(title.strip().lower())
    print(f"Fetching {len(golden_titles)} golden titles from TMDB...")

    # Step 2: Fetch golden movies from TMDB
    golden_movies = {}
    missing_titles = []
    for idx, title in enumerate(golden_titles, 1):
        msg = f"  [{idx}/{len(golden_titles)}] Searching for: {title}"
        print("\r" + msg + " " * (80 - len(msg)), end="")
        results = search_movie_by_title(title)
        best = None
        for r in results:
            r_title = r.get("title", "").strip().lower()
            if r_title == title:
                best = r
                break
        if not best and results:
            best = results[0]
        if not best:
            missing_titles.append(title)
            continue
        try:
            details = get_movie_details(best["id"], language)
            imdb_id = details.get("external_ids", {}).get("imdb_id")
            long_plot = None
            if imdb_id:
                long_plot = fetch_full_plot_by_imdb_id(imdb_id)
            if not long_plot:
                long_plot = fetch_full_plot_by_title(details["title"])
            movie = normalize_movie_from_tmdb(details, enriched_description=long_plot)
            golden_movies[movie["id"]] = movie
        except Exception:
            missing_titles.append(title)
    print(f"\nGolden titles fetched: {len(golden_movies)}")
    if missing_titles:
        raise RuntimeError(f"Missing TMDB entries for golden titles: {missing_titles}")

    # Step 3: Fetch additional movies for the dataset
    sampled_movies = {}
    page = 1
    print(f"Sampling additional movies from TMDB (limit={limit})...")
    while len(sampled_movies) + len(golden_movies) < limit:
        print(f"  Fetching page {page}... Sampled: {len(sampled_movies)}", end="\r")
        if page % 2 == 1:
            resp = get_popular_movies(page, language)
        else:
            resp = get_top_rated_movies(page, language)
        for m in resp.get("results", []):
            mid = m.get("id")
            if mid in golden_movies or mid in sampled_movies:
                continue
            try:
                details = get_movie_details(mid, language)
                imdb_id = details.get("external_ids", {}).get("imdb_id")
                long_plot = None
                if imdb_id:
                    long_plot = fetch_full_plot_by_imdb_id(imdb_id)
                if not long_plot:
                    long_plot = fetch_full_plot_by_title(details["title"])
                movie = normalize_movie_from_tmdb(details, enriched_description=long_plot)
                sampled_movies[movie["id"]] = movie
            except Exception:
                continue
            if len(sampled_movies) + len(golden_movies) >= limit:
                break
        page += 1
    print(f"\nSampled movies fetched: {len(sampled_movies)}")

    # Step 4: Merge and write
    all_movies = list(golden_movies.values()) + list(sampled_movies.values())
    all_movies.sort(key=lambda m: (m["title"].lower(), m["id"]))
    result = {"movies": all_movies}
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=5000)
    parser.add_argument("--language", type=str, default="en-US")
    parser.add_argument("--omdb-only", action="store_true", help="Only use cached OMDb plots, do not make new OMDb requests")
    parser.add_argument("--omdb-max-requests", type=int, default=1000, help="Maximum OMDb requests per run")
    args = parser.parse_args()

    # Set OMDb flags for this run
    if args.omdb_only:
        os.environ["OMDB_ONLY_MODE"] = "1"
    else:
        os.environ["OMDB_ONLY_MODE"] = "0"
    os.environ["OMDB_REQUEST_LIMIT"] = str(args.omdb_max_requests)

    result = build_movies_dataset(args.limit, args.language)
    with open(OUTPUT_PATH, "w") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    print(f"Golden titles: {len(result['movies']) - min(args.limit, len(result['movies']))}")
    print(f"Sampled titles: {min(args.limit, len(result['movies']))}")
    print(f"Total movies: {len(result['movies'])}")

    # OMDb stats
    from movies.omdb_client import omdb_requests_made, omdb_unauthorized
    print(f"OMDb requests made: {omdb_requests_made()}")
    if omdb_unauthorized():
        print("OMDb returned 401 Unauthorized; further requests were skipped.")

if __name__ == "__main__":
    main()
