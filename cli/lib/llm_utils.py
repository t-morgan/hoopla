import os
import logging

from dotenv import load_dotenv
import google.genai as genai
from google.genai import errors as genai_errors


load_dotenv()
logger = logging.getLogger(__name__)


def execute_llm_prompt(
    prompt: str,
    *,
    max_retries: int = 5,
    base_delay: float = 1.0,
) -> str:
    """
    Execute a prompt using Google GenAI (Gemini) and return the response text.

    Retries on rate limit / transient server errors with exponential backoff.
    """
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("GEMINI_API_KEY environment variable is not set.")

    client = genai.Client(api_key=api_key)
    model = os.getenv("GENAI_MODEL", "gemini-2.5-flash")

    def _is_retryable_client_error(e: genai_errors.ClientError) -> bool:
        # Status code is usually available (e.g. 429, 500, 503)
        status = getattr(e, "status_code", None)
        if status in (429, 500, 503):
            return True

        # Fallback: check error payload for RESOURCE_EXHAUSTED
        try:
            payload = getattr(e, "response", None) or {}
            error = payload.get("error") or {}
            if error.get("status") == "RESOURCE_EXHAUSTED":
                return True
        except Exception: # noqa: BLE001
            pass

        return False

    for attempt in range(max_retries + 1):
        try:
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
            return text

        except genai_errors.ClientError as e:
            if attempt < max_retries and _is_retryable_client_error(e):
                # Exponential backoff with jitter
                sleep_for = base_delay * (2 ** attempt) + random.uniform(0, base_delay)
                logger.warning(
                    "LLM rate/availability error (attempt %s/%s): %s; "
                    "retrying in %.1fs",
                    attempt + 1,
                    max_retries,
                    e,
                    sleep_for,
                )
                time.sleep(sleep_for)
                continue

            logger.warning("Non-retryable LLM client error: %s", e)
            return ""

        except Exception as e:  # noqa: BLE001
            logger.warning("LLM error: %s", e)
            return ""

    # If we exhausted retries without returning
    logger.warning("LLM: exhausted retries, returning empty string")
    return ""