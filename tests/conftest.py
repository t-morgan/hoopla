# Ensure project root is on sys.path so tests can import top-level packages like `cli`
import sys
from pathlib import Path
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from cli.lib.search_utils import load_movies


@pytest.fixture(scope="session")
def actor_search_tool():
    """Shared ActorSearchTool instance for all tests.

    Uses session scope to cache the expensive imports (sentence_transformers, google.genai)
    across all tests. First test takes ~20s, subsequent tests are fast.
    """
    from cli.lib.agentic_rag import ActorSearchTool

    movies = load_movies()
    return ActorSearchTool(movies)
