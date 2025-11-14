import os
import logging
import random
import time
from typing import Sequence, Any, Optional, Union

from dotenv import load_dotenv
import google.genai as genai
from google.genai import errors as genai_errors


load_dotenv()
logger = logging.getLogger(__name__)


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

    for attempt in range(max_retries + 1):
        try:
            resp = client.models.generate_content(model=model, contents=contents)
            # Attach a convenience normalized_text attribute (non-invasive)
            try:
                text = (resp.text or "").strip()
                if (text.startswith("`") and text.endswith("`")) or (
                    text.startswith("```") and text.endswith("```")
                ):
                    text = text.strip("`")
                if (text.startswith('"') and text.endswith('"')) or (
                    text.startswith("'") and text.endswith("'")
                ):
                    text = text[1:-1].strip()
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
    return text

