# Agentic Search Tools

This directory contains modular search tool implementations used by the AgenticRAG system. Each tool provides a specific search capability that the LLM-powered agent can dynamically select and combine based on the user's query.

## Architecture

All tools inherit from the `SearchTool` abstract base class defined in `base.py`:

```python
class SearchTool(ABC):
    def __init__(self, name: str, description: str):
        self.name = name
        self.description = description
    
    @abstractmethod
    def search(self, query: str, limit: int = 10) -> list[dict[str, Any]]:
        """Execute the search and return results."""
        pass
```

## Available Tools

### KeywordSearchTool (`keyword_search_tool.py`)
- **Name**: `keyword_search`
- **Description**: BM25-based keyword search
- **Best for**: Exact term matching, specific keywords
- **Algorithm**: Uses inverted index with BM25 scoring

### SemanticSearchTool (`semantic_search_tool.py`)
- **Name**: `semantic_search`
- **Description**: Embedding-based semantic search
- **Best for**: Conceptual queries, finding similar meanings
- **Algorithm**: Cosine similarity on text embeddings

### HybridSearchTool (`hybrid_search_tool.py`)
- **Name**: `hybrid_search`
- **Description**: Reciprocal Rank Fusion of keyword + semantic
- **Best for**: Balanced search combining exact matches and concepts
- **Algorithm**: RRF merging of keyword and semantic results

### RegexSearchTool (`regex_search_tool.py`)
- **Name**: `regex_search`
- **Description**: Pattern matching with regular expressions
- **Best for**: Exact phrases, specific patterns
- **Algorithm**: Regex matching on title and description

### GenreSearchTool (`genre_search_tool.py`)
- **Name**: `genre_search`
- **Description**: Genre filtering with synonym mapping
- **Best for**: Finding movies by genre (horror, comedy, action, etc.)
- **Features**:
  - Synonym mapping (e.g., "suspense" → "thriller")
  - Metadata genre matching
  - Text-based fallback
  - Configurable scoring

### ActorSearchTool (`actor_search_tool.py`)
- **Name**: `actor_search`
- **Description**: Actor name matching with normalization
- **Best for**: Finding movies by actor/cast
- **Features**:
  - Name normalization (handles punctuation, case, titles)
  - BM25 for initial recall
  - Name-based re-ranking (full name, last name, fuzzy match)
  - Minimum confidence threshold filtering

## Shared Components

### `base.py`
Abstract base class `SearchTool` that all tools inherit from.

### `constants.py`
Shared constants including:
- `GENRE_SYNONYMS`: Mapping of genre synonyms to canonical names

### `utils.py`
Shared utility functions:
- `extract_json_object()`: Extracts JSON from LLM responses

## Creating a New Tool

To add a new search tool:

1. **Create a new file** in this directory (e.g., `director_search_tool.py`)

2. **Implement the SearchTool interface**:
   ```python
   from typing import Any
   from .base import SearchTool
   from .search_utils import load_movies  # or your data loading method
   
   class DirectorSearchTool(SearchTool):
       def __init__(self, movies: list[dict[str, Any]] | None = None):
           super().__init__(
               name="director_search",
               description="Search for movies by director name"
           )
           self.movies = movies or load_movies()
       
       def search(self, query: str, limit: int = 10) -> list[dict[str, Any]]:
           # Implement your search logic
           results = []
           query_lower = query.lower()
           
           for movie in self.movies:
               director = movie.get('director', '').lower()
               if query_lower in director:
                   results.append({
                       **movie,
                       'score': 1.0 if director == query_lower else 0.8
                   })
           
           # Sort by score and limit
           results.sort(key=lambda x: x['score'], reverse=True)
           return results[:limit]
   ```

3. **Export in `__init__.py`**:
   ```python
   from .director_search_tool import DirectorSearchTool
   
   __all__ = [
       # ...existing exports...
       'DirectorSearchTool',
   ]
   ```

4. **Register in AgenticRAG** (`cli/lib/agentic_rag.py`):
   ```python
   from .agentic_tools import (
       # ...existing imports...
       DirectorSearchTool,
   )
   
   class AgenticRAG:
       def __init__(self, ...):
           # ...
           self.tools = {
               # ...existing tools...
               'director_search': DirectorSearchTool(self.movies),
           }
   ```

The LLM agent will automatically discover and use your tool based on its description!

## Tool Selection Process

The AgenticRAG system uses an LLM to dynamically select tools:

1. **Context Building**: The LLM receives:
   - Original user query
   - Available tools with descriptions
   - Search history (previous tools/queries/results)
   - Current candidate pool
   - Already-tried tool+query combinations

2. **Decision**: The LLM decides:
   - Whether to continue searching
   - Which tool to use next
   - What query to pass to that tool
   - Reasoning for the choice

3. **Execution**: The selected tool is executed with the refined query

4. **Iteration**: Results are merged and the process repeats until:
   - Sufficient results are found
   - Max iterations reached
   - No new tool+query combinations are useful

## Testing Tools

Tools should be tested both in isolation and as part of the agentic system:

```python
# Unit test a specific tool
from cli.lib.agentic_tools import ActorSearchTool

tool = ActorSearchTool()
results = tool.search("Tom Hanks", limit=5)
assert len(results) > 0
assert "Tom Hanks" in results[0]['cast']

# Integration test with AgenticRAG
from cli.lib.agentic_rag import AgenticRAG, AgenticSearchConfig

config = AgenticSearchConfig(max_iterations=3, debug=True)
agent = AgenticRAG(config)
result = agent.search("Tom Hanks horror movies")
# Should use both actor_search and genre_search tools
```

## Best Practices

1. **Clear Descriptions**: Write tool descriptions that help the LLM choose appropriately
2. **Consistent Interface**: Always return a list of movie dicts with at least `id` and `score` fields
3. **Score Normalization**: Keep scores in the 0.0-1.0 range for consistent merging
4. **Error Handling**: Handle edge cases gracefully (empty queries, missing data, etc.)
5. **Performance**: Consider caching for expensive operations (embeddings, indexes)
6. **Documentation**: Document the tool's algorithm, best use cases, and limitations

## Dependencies

Tools may depend on:
- `cli/lib/search_utils.py` - Data loading utilities
- `cli/lib/inverted_index.py` - BM25 keyword search
- `cli/lib/semantic_search.py` - Embedding-based search
- `cli/lib/hybrid_search.py` - RRF fusion
- External libraries: `numpy`, `scikit-learn`, `google.generativeai`

See `pyproject.toml` for the full dependency list.

