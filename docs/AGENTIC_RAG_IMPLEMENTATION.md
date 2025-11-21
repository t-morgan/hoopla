# Agentic Recursive RAG Implementation

## Overview

An agent that dynamically chooses and combines search tools (keyword, semantic, hybrid, regex, genre, actor) based on the query and prior results. It iterates until sufficient results are found, merges them (union or intersection), and can generate an answer with citations. Results are optionally re-ranked by an LLM for final relevance.

## Architecture

- `AgenticRAG` (`cli/lib/agentic_rag.py`) orchestrates the loop
- Search tools (modular implementations in `cli/lib/agentic_tools/`):
  - `KeywordSearchTool` (`keyword_search_tool.py`) - BM25 keyword search
  - `SemanticSearchTool` (`semantic_search_tool.py`) - Embedding-based search
  - `HybridSearchTool` (`hybrid_search_tool.py`) - RRF fusion of keyword + semantic
  - `RegexSearchTool` (`regex_search_tool.py`) - Pattern matching
  - `GenreSearchTool` (`genre_search_tool.py`) - Genre filtering with synonym mapping
  - `ActorSearchTool` (`actor_search_tool.py`) - Actor name matching with normalization
  - `SearchTool` (`base.py`) - Abstract base class for all tools
  - Shared utilities (`utils.py`) - JSON extraction and common helpers
  - Constants (`constants.py`) - Genre synonyms and other shared data
- CLI: `cli/agentic_rag_cli.py` with `search` and `generate`

### Tool Architecture

Each tool inherits from `SearchTool` base class and implements:
- `name`: Tool identifier
- `description`: Used by the LLM for tool selection
- `search(query, limit)`: Main search method returning list of movie dicts

## Agentic Loop (high level)

```text
while searching:
    decision = pick_next_tool(history, candidate_pool)
    if not decision:
        break
    results = run_tool(decision)
    history.append(results)
    candidate_pool.update(results)
merge_and_return(history)
rerank_with_llm()
```

Guidelines:
- Use actor_search FIRST when actors are mentioned in the query
- Use genre_search SECOND to filter/refine actor results (e.g., "horror movies with Tom Hanks")
- For queries with both actors AND genres/keywords, search for BOTH to find intersection
- Use semantic/hybrid for general concept searches without specific actors
- Use regex_search for exact phrase matching
- Stop after finding 2-3 complementary searches or when enough results found
- When combining filters (actor + genre), both searches will be intersected automatically

## Tool Selection

- The agent uses an LLM to select the next tool and query, based on:
  - Search history (previous tool/query/results)
  - Candidate pool (top results so far)
  - Tool descriptions
  - Already tried tool+query pairs (to avoid repeats)
- The LLM receives a prompt with all this context and responds with JSON specifying:
  - Whether to continue searching
  - Which tool to use next
  - What query to use
  - Reasoning for the choice

## Merge Strategies

- **Union:** Combines all unique results from similar tools (semantic, keyword, hybrid)
- **Intersection:** Only keeps movies found by multiple tools (e.g., actor + genre)
- **Auto:** Uses intersection for complementary tools (actor+genre), union otherwise
- **Fallback:** If intersection returns zero results, falls back to weighted union (actor/genre results weighted higher)

## Genre and Actor Search Details

- **GenreSearchTool:**
  - Uses a synonym → canonical genre mapping (e.g., "suspense" → "thriller")
  - Matches genres via explicit metadata or text synonyms
  - Scores movies higher for metadata matches, lower for text-only matches
- **ActorSearchTool:**
  - Normalizes actor names (removes punctuation, lowercases, strips filler words)
  - Uses BM25 for recall, then re-ranks by name strength (full name, last name, fuzzy match)
  - Only returns movies where actor match strength is high enough

## LLM-Based Reranking

- After merging results, the agent can re-rank movies using an LLM prompt
- The LLM receives the user query and a compact list of candidate movies with scores
- It returns a new ranking by relevance, which is used for the final result order

## Usage

**Important:** Always run CLI and scripts with `PYTHONPATH=src` to ensure imports work with the src layout.

CLI:

```bash
PYTHONPATH=src uv run python src/cli/agentic_rag_cli.py search "your query"
PYTHONPATH=src uv run python src/cli/agentic_rag_cli.py generate "your question"
```

> If you see `ModuleNotFoundError: No module named 'movies'`, you likely forgot to set `PYTHONPATH=src`.

Python API:

```python
from cli.lib.agentic_rag import AgenticRAG, AgenticSearchConfig

config = AgenticSearchConfig(max_iterations=5, final_result_limit=5, debug=False)
agent = AgenticRAG(config)
results = agent.search("scary bear movies")
answer = agent.search_and_generate("find action movies with bears")
```

## Example Strategies

- Genre + theme: genre_search → regex_search("bear") → semantic_search
- Actor + theme: actor_search → semantic_search → hybrid_search
- Complex: hybrid_search → regex_search → semantic_search

## Future Enhancements (brief)

- Additional tools (year, rating, director, multimodal)
- Smarter result fusion and caching
- Parallel execution and diversity optimization

## Extending with New Tools

To add a new search tool:

1. Create a new file in `cli/lib/agentic_tools/` (e.g., `my_tool.py`)
2. Inherit from `SearchTool` base class:
   ```python
   from .base import SearchTool
   
   class MySearchTool(SearchTool):
       def __init__(self):
           super().__init__(
               name="my_search",
               description="Description for LLM tool selection"
           )
       
       def search(self, query: str, limit: int = 10) -> list[dict[str, Any]]:
           # Implement your search logic
           return results
   ```
3. Export it in `__init__.py`:
   ```python
   from .my_tool import MySearchTool
   __all__ = [..., 'MySearchTool']
   ```
4. Register it in `AgenticRAG.__init__()`:
   ```python
   self.tools = {
       # ...existing tools...
       'my_search': MySearchTool(),
   }
   ```

The LLM will automatically discover and use your tool based on its description.
