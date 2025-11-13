import json
import os
import logging
import google.genai as genai
from dotenv import load_dotenv
load_dotenv()

BM25_B = 0.75
BM25_K1 = 1.5
DEFAULT_SEARCH_LIMIT = 5

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
DATA_PATH = os.path.join(PROJECT_ROOT, "data", "movies.json")
STOPWORDS_PATH = os.path.join(PROJECT_ROOT, "data", "stopwords.txt")

logger = logging.getLogger(__name__)


def load_movies() -> list[dict]:
    with open(DATA_PATH, "r") as f:
        data = json.load(f)
    return data["movies"]


def load_stopwords() -> list[str]:
    with open(STOPWORDS_PATH, "r") as f:
        return f.read().splitlines()


def _spell_correct_with_llm(query: str) -> str:
    """
    Attempt to correct spelling in a short search query using Google GenAI (Gemini).
    Falls back to the original query if the API is not configured or any error occurs.
    """
    query = query.strip()
    if not query:
        return query

    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        return query
    try:
        client = genai.Client(api_key=api_key)
        # Use a fast, low-cost model suitable for lightweight text transforms.
        model = os.getenv("GENAI_MODEL", "gemini-2.5-flash")
        prompt = f"""\
You correct spelling mistakes in short movie search queries.

Rules:
- Correct misspellings, including proper nouns (movie titles, characters, actors, places).
- Prefer correcting to the nearest valid English spelling.
- Do not add or remove words.
- Return ONLY the corrected query text.
- If the query is already correct, return it unchanged.
- The domain is movies and entertainment.

Query: "{query}"
 """
        resp = client.models.generate_content(model=model, contents=prompt)
        text = (resp.text or "").strip()
        # Normalize simple wrappers users/models sometimes add
        if (text.startswith("`") and text.endswith("`")) or (
            text.startswith("```") and text.endswith("```")
        ):
            text = text.strip("`")
        if (text.startswith('"') and text.endswith('"')) or (
            text.startswith("'") and text.endswith("'")
        ):
            text = text[1:-1].strip()
        return text or query
    except Exception as e:  # noqa: BLE001
        logger.warning("LLM spell correction error: %s", e)
        return query


def enhance_query(query: str, method: str | None) -> str:
    """
    Enhance the query text according to the requested method.
    Currently supports:
      - "spell": LLM-powered spelling correction.
    If method is None or unrecognized, returns the original query.
    """
    if method == "spell":
        enhanced_query = _spell_correct_with_llm(query)
        print(f"Enhanced query ({method}): '{query}' -> '{enhanced_query}'\n")
        return enhanced_query
    return query
