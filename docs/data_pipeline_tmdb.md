# TMDB Data Pipeline for Movie Search

## Overview

`data/movies.json` is generated automatically from The Movie Database (TMDB) API. This ensures a consistent, up-to-date movie dataset for search and retrieval.

## Project Layout

- All core code is under `src/` (e.g., `src/movies/`, `src/cli/`)
- Scripts are in `scripts/`
- Tests are in `tests/`
- Data files are in `data/`

## Environment Setup

- Store your TMDB API key in a `.env` file or as an environment variable:
  ```sh
  TMDB_API_KEY=your_api_key_here
  ```
- The pipeline loads `.env` automatically using `python-dotenv`.

## Movie Schema

Each movie entry in `data/movies.json` has the following shape:

```
{
  "id": 123,
  "title": "Paddington",
  "description": "A polite bear in London...",
  "cast": ["Ben Whishaw", "Hugh Bonneville", "Sally Hawkins"],
  "genre": ["Comedy", "Family", "Adventure"]
}
```
- **genre** is a list of genre names from TMDB (e.g., `["Comedy", "Family", "Adventure"]`).
- Some movies may have multiple genres; some may have an empty list (`[]`).
- Downstream components (search, embeddings, etc.) treat all genre entries as part of the searchable text.

## Building the Dataset

Run the pipeline to generate `data/movies.json`:

```sh
PYTHONPATH=src uv run scripts/build_movies_json.py --limit 5000 --language en-US
```
- `--limit`: Number of movies to sample (excluding golden titles)
- `--language`: Language for TMDB queries (default: en-US)

## Golden Guarantee

All movie titles listed in `data/golden_dataset.json` (`relevant_docs`) are always included in the output. If any cannot be found via TMDB, the build fails with a clear error.

## Testing

- Tests mock TMDB API calls and do **not** hit the real TMDB service.
- You can run all tests with:
  ```sh
  PYTHONPATH=src uv run pytest
  ```

## OMDb Enrichment Modes

- By default, the pipeline will attempt to enrich movie descriptions with OMDb full plots, up to the OMDb API daily limit (default: 1000 requests).
- Use `--omdb-only` to run in cache-only mode: no new OMDb requests are made, only cached plots are used for enrichment.
- Use `--omdb-max-requests N` to set a custom OMDb request limit per run.
- If OMDb returns 401 Unauthorized, further OMDb requests are skipped for the run.

### Example Usage

```sh
# Build with OMDb enrichment (up to 1000 requests)
PYTHONPATH=src uv run scripts/build_movies_json.py --limit 5000 --language en-US

# Build using only cached OMDb plots (no new OMDb requests)
PYTHONPATH=src uv run scripts/build_movies_json.py --limit 5000 --language en-US --omdb-only

# Build with a custom OMDb request limit
PYTHONPATH=src uv run scripts/build_movies_json.py --limit 5000 --language en-US --omdb-max-requests 200
```

- The pipeline will always produce a complete movies.json, using TMDB overview for any movie not yet enriched by OMDb.
- OMDb enrichment is incremental: run the pipeline over multiple days to gradually fill the cache and maximize OMDb coverage.

## Example Workflow

```sh
# Set your API key
export TMDB_API_KEY=your_api_key_here

# Build the dataset
PYTHONPATH=src uv run scripts/build_movies_json.py --limit 5000 --language en-US

# Run tests
PYTHONPATH=src uv run pytest
```

---
For more details, see the implementation in `src/` and `scripts/`.
