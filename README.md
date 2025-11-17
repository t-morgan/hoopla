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

Run a search:

```bash
uv run python cli/agentic_rag_cli.py search "scary bear movies in the forest"
```

Generate an answer:

```bash
uv run python cli/agentic_rag_cli.py generate "find action movies with bears"
```

With options:

```bash
uv run python cli/agentic_rag_cli.py search "leonardo dicaprio" \
  --max-iterations 5 \
  --max-results 10 \
  --limit 5 \
  --debug
```

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
cli/
├── agentic_rag_cli.py          # CLI entrypoint
├── lib/
│   ├── agentic_rag.py          # Main AgenticRAG orchestrator
│   └── agentic_tools/          # Modular search tool implementations
│       ├── __init__.py         # Tool exports
│       ├── base.py             # SearchTool abstract base class
│       ├── constants.py        # Shared constants (genre synonyms, etc.)
│       ├── utils.py            # Shared utilities
│       ├── keyword_search_tool.py
│       ├── semantic_search_tool.py
│       ├── hybrid_search_tool.py
│       ├── regex_search_tool.py
│       ├── genre_search_tool.py
│       └── actor_search_tool.py
data/
├── movies.json                 # Movie dataset
└── golden_dataset.json         # Test/evaluation data
docs/
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
