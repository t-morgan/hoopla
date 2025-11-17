import os
import logging
import random
import time
from typing import Sequence, Any, Optional, Union

from dotenv import load_dotenv
import google.genai as genai
from google.genai import errors as genai_errors

# Provide fallback attributes for stubbed google.genai in tests so monkeypatch can replace them
try:
    _ = getattr(genai, "Client")
except Exception:  # noqa: BLE001
    class _StubClient:  # minimal placeholder; tests will monkeypatch over this
        def __init__(self, *args, **kwargs):
            raise RuntimeError("google.genai.Client is not available in this environment")
    try:
        setattr(genai, "Client", _StubClient)
    except Exception:
        pass

try:
    _ = getattr(genai_errors, "ClientError")
except Exception:  # noqa: BLE001
    class _StubClientError(Exception):
        pass
    try:
        setattr(genai_errors, "ClientError", _StubClientError)
    except Exception:
        pass


load_dotenv()
logger = logging.getLogger(__name__)


def normalize_llm_text(text: Optional[str]) -> str:
    """Normalize raw LLM text output.

    - Strips surrounding whitespace and BOM
    - Unwraps triple backtick code fences, including optional language tag
      like ```json, ```python, etc.
    - Removes a standalone leading language tag line (e.g., "json\n")
    - Removes surrounding single/double quotes if the entire content is quoted
    - Returns a clean string suitable for JSON parsing when applicable
    """
    if not text:
        return ""
    # Normalize whitespace and BOM
    t: str = str(text).strip().lstrip("\ufeff").rstrip("\ufeff").strip()

    # Fast path: if fenced, extract the inner content via regex
    try:
        import re
        fenced = re.match(r"^```([a-zA-Z0-9_+-]*)\s*\n([\s\S]*?)\n?```\s*$", t)
        if fenced:
            lang = (fenced.group(1) or "").strip().lower()
            inner = fenced.group(2)
            t = inner
            # If first line repeats the language tag (e.g., 'json'), drop it
            first_nl = t.find("\n")
            if first_nl != -1:
                first_line = t[:first_nl].strip().lower()
                if first_line in {"json", lang} and len(t[first_nl + 1 :].strip()) > 0:
                    t = t[first_nl + 1 :]
        else:
            # Handle degenerate case where content starts/ends with backticks but not matched by regex
            if (t.startswith("```") and t.endswith("```") ) or (t.startswith("`") and t.endswith("`")):
                t = t.strip("`")
    except Exception:  # noqa: BLE001
        # Fall back to basic stripping of backticks if regex fails
        if (t.startswith("```") and t.endswith("```")) or (t.startswith("`") and t.endswith("`")):
            t = t.strip("`")

    # Remove a leading language tag token if present (e.g., "json\n{...}")
    if t.lower().startswith("json\n"):
        t = t[len("json\n"):]

    # Remove surrounding quotes if the entire content is quoted
    if (t.startswith('"') and t.endswith('"')) or (t.startswith("'") and t.endswith("'")):
        t = t[1:-1]

    return t.strip()


def execute_llm_response(
    prompt: Optional[str] = None,
    *,
    parts: Optional[Sequence[Any]] = None,
    max_retries: int = 5,
    base_delay: float = 1.0,
) -> Optional[Any]:
    """Execute a Gemini request and return the full response object.

    Mirrors execute_llm_prompt but returns the raw google-genai response
    (GenerateContentResponse) instead of just the normalized text.

    Inputs:
        prompt: Single string prompt (ignored if parts provided)
        parts: Sequence of content parts (strings or structured parts)
        max_retries: Retry attempts for transient errors
        base_delay: Base delay used in exponential backoff

    Returns:
        google.genai.types.GenerateContentResponse on success, or None on failure.

    Raises:
        ValueError for missing inputs or missing API key.
    """
    if prompt is None and (parts is None or len(parts) == 0):
        raise ValueError("Either 'prompt' or non-empty 'parts' must be provided.")

    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("GEMINI_API_KEY environment variable is not set.")

    client = genai.Client(api_key=api_key)
    model = os.getenv("GENAI_MODEL", "gemini-2.5-flash")

    contents: Union[str, Sequence[Any]]
    if parts is not None:
        contents = [p for p in parts if p is not None]
        if not contents:
            raise ValueError("'parts' provided but empty after filtering None values.")
    else:
        contents = prompt

    def _is_retryable_client_error(e: genai_errors.ClientError) -> bool:
        status = getattr(e, "status_code", None)
        if status in (429, 500, 503):
            return True
        try:
            payload = getattr(e, "response", None)
            if isinstance(payload, dict):
                error = payload.get("error") or {}
                if error.get("status") == "RESOURCE_EXHAUSTED":
                    return True
            elif hasattr(payload, "error"):
                error = getattr(payload, "error", None)
                if isinstance(error, dict) and error.get("status") == "RESOURCE_EXHAUSTED":
                    return True
        except Exception as ex:  # noqa: BLE001
            logger.error("Error checking retryable client error: %s", ex)
        return False

    for attempt in range(int(max_retries) + 1):
        try:
            resp = client.models.generate_content(model=model, contents=contents)
            # Attach a convenience normalized_text attribute (non-invasive)
            try:
                raw = (getattr(resp, "text", "") or "")
                text = normalize_llm_text(raw)
                setattr(resp, "normalized_text", text)
            except Exception:  # noqa: BLE001
                pass
            return resp
        except genai_errors.ClientError as e:
            if attempt < max_retries and _is_retryable_client_error(e):
                sleep_for = base_delay * (2 ** attempt) + random.uniform(0, base_delay)
                logger.warning(
                    "LLM rate/availability error (attempt %s/%s): %s; retrying in %.1fs",
                    attempt + 1,
                    max_retries,
                    e,
                    sleep_for,
                )
                time.sleep(sleep_for)
                continue
            logger.warning("Non-retryable LLM client error: %s", e)
            return None
        except Exception as e:  # noqa: BLE001
            error_str = str(e).lower()
            is_retryable = any(x in error_str for x in ['503', '429', '500', 'unavailable', 'overloaded', 'resource_exhausted'])
            if attempt < max_retries and is_retryable:
                sleep_for = base_delay * (2 ** attempt) + random.uniform(0, base_delay)
                logger.warning(
                    "LLM transient error (attempt %s/%s): %s; retrying in %.1fs",
                    attempt + 1,
                    max_retries,
                    e,
                    sleep_for,
                )
                time.sleep(sleep_for)
                continue
            logger.error("LLM error: %s", e)
            return None

    logger.warning("LLM: exhausted retries, returning None")
    return None


def execute_llm_prompt(
    prompt: Optional[str] = None,
    *,
    parts: Optional[Sequence[Any]] = None,
    max_retries: int = 5,
    base_delay: float = 1.0,
) -> str:
    """Thin wrapper returning just normalized text.

    Delegates all request/retry logic to execute_llm_response.
    """
    # Input validation (kept for backward compatibility)
    if prompt is None and (parts is None or len(parts) == 0):
        raise ValueError("Either 'prompt' or non-empty 'parts' must be provided.")

    resp = execute_llm_response(
        prompt=prompt,
        parts=parts,
        max_retries=max_retries,
        base_delay=base_delay,
    )
    if resp is None:
        return ""
    # Prefer normalized_text set by execute_llm_response; fallback to .text
    text = getattr(resp, "normalized_text", None) or (getattr(resp, "text", "") or "")
    # Ensure final pass of normalization in case caller bypassed response object
    return normalize_llm_text(text)
