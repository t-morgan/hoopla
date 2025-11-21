"""Utility functions for agentic search tools."""

import json
import logging

from ..llm_utils import normalize_llm_text

logger = logging.getLogger(__name__)


def extract_json_object(maybe_text: str) -> dict | None:
    """Extract a JSON object from text that may include code fences or noise.

    Strategy:
    - First normalize text (strip code fences and quotes)
    - Try direct json.loads
    - If it fails, search for the first top-level {...} block via brace counting
    """
    if not maybe_text:
        return None
    cleaned = normalize_llm_text(maybe_text)

    # Quick path
    try:
        return json.loads(cleaned)
    except Exception as e:
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(f"Failed to parse JSON directly. First 500 chars: {cleaned[:500]}")
        pass

    # Fallback: find the first top-level JSON object in the string
    start = cleaned.find('{')
    if start == -1:
        return None
    depth = 0
    end = None
    for i, ch in enumerate(cleaned[start:], start=start):
        if ch == '{':
            depth += 1
        elif ch == '}':
            depth -= 1
            if depth == 0:
                end = i + 1
                break
    if end is None:
        return None
    snippet = cleaned[start:end]
    try:
        return json.loads(snippet)
    except Exception as e:
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(f"Failed to parse JSON from snippet. First 500 chars: {snippet[:500]}")
        return None

