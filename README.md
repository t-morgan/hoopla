# Hoopla Agentic RAG

An agentic, iterative retrieval-and-generation system for movie search on Hoopla. The agent dynamically selects among multiple search tools (keyword, semantic, hybrid, regex, genre, actor), merges results, and can generate a natural-language answer with citations.

## Prerequisites
- Python 3.11+
- uv (recommended) or pip
- A Gemini API key for LLM-powered steps
  - Set GEMINI_API_KEY in your environment (e.g., via a .env file)

## Installation

Using uv (recommended):

```bash
uv venv
uv pip install -e .
```

Using pip:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

## Quickstart

**Important:** Always run CLI and scripts with `PYTHONPATH=src` to ensure imports work with the src layout.

Run a search:

```bash
PYTHONPATH=src uv run python src/cli/agentic_rag_cli.py search "scary bear movies in the forest"
```

Generate an answer:

```bash
PYTHONPATH=src uv run python src/cli/agentic_rag_cli.py generate "find action movies with bears"
```

With options:

```bash
PYTHONPATH=src uv run python src/cli/agentic_rag_cli.py search "leonardo dicaprio" \
  --max-iterations 5 \
  --max-results 10 \
  --limit 5 \
  --debug
```

> If you see `ModuleNotFoundError: No module named 'movies'`, you likely forgot to set `PYTHONPATH=src`.

## Python API

```python
from cli.lib.agentic_rag import AgenticRAG, AgenticSearchConfig

config = AgenticSearchConfig(max_iterations=5, final_result_limit=5, debug=True)
agent = AgenticRAG(config)

result = agent.search("scary bear movies")
answer = agent.search_and_generate("find action movies with bears")
```

## Project Structure

```
src/
├── cli/
│   ├── agentic_rag_cli.py          # CLI entrypoint
│   └── lib/
│       ├── agentic_rag.py          # Main AgenticRAG orchestrator
│       └── agentic_tools/          # Modular search tool implementations
│           ├── __init__.py         # Tool exports
│           ├── base.py             # SearchTool abstract base class
│           ├── constants.py        # Shared constants (genre synonyms, etc.)
│           ├── utils.py            # Shared utilities
│           ├── keyword_search_tool.py
│           ├── semantic_search_tool.py
│           ├── hybrid_search_tool.py
│           ├── regex_search_tool.py
│           ├── genre_search_tool.py
│           └── actor_search_tool.py
├── data/
│   ├── movies.json                 # Movie dataset
│   └── golden_dataset.json         # Test/evaluation data
└── docs/
    └── AGENTIC_RAG_IMPLEMENTATION.md  # Detailed architecture docs
```

## Documentation
- [Agentic RAG Implementation](docs/AGENTIC_RAG_IMPLEMENTATION.md): Architecture and agentic loop details

## Contributing & Environment Setup
Always use `uv run` for all commands to ensure dependencies are available. See above for installation and environment setup.

## Environment
Create a `.env` or export environment variables:

```bash
export GEMINI_API_KEY=your_key_here
# Optional: override default model
export GENAI_MODEL=gemini-2.5-flash
```

## Testing

Run tests with pytest (must use `uv run`):

```bash
uv run pytest
```

Or use the Makefile:

```bash
make test          # Run all tests
make test-fast     # Run only fast unit tests
make test-slow     # Run only slow integration tests
```

**Important:** Always use `uv run pytest`, not just `pytest`, to ensure dependencies are available.

## License
TBD

# Hoopla Movie Search System

## Project Layout

- All core code is under `src/`:
  - `src/movies/` (TMDB client, normalization, etc.)
  - `src/cli/` (CLI tools)
  - `src/build/` (build utilities)
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

Each movie in `data/movies.json` has the following shape:

```
{
  "id": 123,
  "title": "Paddington",
  "description": "A polite bear in London... (OMDb full plot if available, otherwise TMDB overview)",
  "cast": ["Ben Whishaw", "Hugh Bonneville", "Sally Hawkins"],
  "genre": ["Comedy", "Family", "Adventure"]
}
```
- `description` is the long-form plot from OMDb (full plot) when available; otherwise, it falls back to TMDB's overview.
- `genre` is a list of genre names from TMDB. Some movies have multiple genres; some may have an empty list.
- All genre entries are included in the searchable text for BM25 and embeddings.

## Building the Dataset

Run the pipeline to generate `data/movies.json`:

```sh
PYTHONPATH=src uv run scripts/build_movies_json.py --limit 5000 --language en-US
```

- `--limit`: Number of movies to sample (excluding golden titles)
- `--language`: Language for TMDB queries (default: en-US)

## Golden Guarantee

All movie titles listed in `data/golden_dataset.json` (`relevant_docs`) are always included in the output. If any cannot be found via TMDB, the build fails with a clear error.

## OMDb Enrichment Modes

- By default, the pipeline will attempt to enrich movie descriptions with OMDb full plots, up to the OMDb API daily limit (default: 1000 requests).
- Use `--omdb-only` to run in cache-only mode: no new OMDb requests are made, only cached plots are used for enrichment.
- Use `--omdb-max-requests N` to set a custom OMDb request limit per run.
- If OMDb returns 401 Unauthorized, further OMDb requests are skipped for the run.

**OMDb enrichment is incremental and cache-based:**
- The pipeline will always produce a complete `movies.json`, using TMDB overview for any movie not yet enriched by OMDb.
- OMDb plots are cached to disk and reused in future runs.
- You can safely run the pipeline multiple times (with or without `--omdb-only`) to gradually fill the cache and maximize OMDb coverage without exceeding daily limits.
- Golden title rules are always enforced: all golden titles are present, and missing TMDB titles will fail the build.

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

## Testing

- Tests mock TMDB API calls and do **not** hit the real TMDB service.
- Run all tests with:
  ```sh
  PYTHONPATH=src uv run pytest
  ```

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
For more details, see `docs/data_pipeline_tmdb.md` and the implementation in `src/` and `scripts/`.
