import os
import types
import sys
import pytest

# Mark all tests in this module as fast unit tests since they use mocking
pytestmark = pytest.mark.unit

# Stub google.genai and google.genai.errors before importing llm_utils
google_mod = types.ModuleType('google')
# Submodule placeholders
genai_mod = types.ModuleType('google.genai')
errors_mod = types.ModuleType('google.genai.errors')
# Ensure parent/child linkage for 'from google.genai import errors as genai_errors'
setattr(genai_mod, 'errors', errors_mod)
# Register in sys.modules
sys.modules.setdefault('google', google_mod)
sys.modules.setdefault('google.genai', genai_mod)
sys.modules.setdefault('google.genai.errors', errors_mod)

# Stub dotenv to avoid dependency requirement in tests
dotenv_mod = types.ModuleType('dotenv')
setattr(dotenv_mod, 'load_dotenv', lambda *a, **k: None)
sys.modules.setdefault('dotenv', dotenv_mod)

# Now import the module under test
import cli.lib.llm_utils as llm_utils


class DummyResponse:
    def __init__(self, text: str | None):
        self.text = text


class DummyModels:
    def __init__(self, generator):
        self._generator = generator

    def generate_content(self, *, model: str, contents):  # mimic keyword-only signature usage
        return self._generator(model=model, contents=contents)


class DummyClient:
    def __init__(self, *, generator):
        self.models = DummyModels(generator)


class TransientError(Exception):
    status_code = 503


def test_normalize_llm_text_variants():
    # Plain passthrough
    assert llm_utils.normalize_llm_text("hello") == "hello"

    # Triple-fenced json
    fenced = """```json\n{\n  \"a\": 1\n}\n```"""
    assert llm_utils.normalize_llm_text(fenced) == '{\n  "a": 1\n}'

    # Single backticks
    assert llm_utils.normalize_llm_text("`quoted`\n") == "quoted"

    # Surrounding quotes
    assert llm_utils.normalize_llm_text('"wrapped"') == "wrapped"
    assert llm_utils.normalize_llm_text("'wrapped'") == "wrapped"


def test_execute_llm_prompt_strips_fences(monkeypatch):
    # Ensure API key present
    monkeypatch.setenv("GEMINI_API_KEY", "dummy")

    # Monkeypatch client and errors
    out_text = """```json\n{\n  \"ok\": true\n}\n```"""

    def gen(**kwargs):
        # Expect contents to be the prompt string
        assert isinstance(kwargs.get("contents"), str)
        return DummyResponse(out_text)

    monkeypatch.setattr(llm_utils.genai, "Client", lambda api_key: DummyClient(generator=gen))

    result = llm_utils.execute_llm_prompt(prompt="return json")
    assert result.strip().startswith("{")
    assert result.strip().endswith("}")


def test_execute_llm_response_retry_then_success(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "dummy")

    # Patch error type so llm_utils catches it as retryable
    class ClientError(Exception):
        status_code = 503
    monkeypatch.setattr(llm_utils.genai_errors, "ClientError", ClientError)

    calls = {"n": 0}

    def gen(**kwargs):
        calls["n"] += 1
        if calls["n"] == 1:
            raise ClientError("temporarily unavailable")
        return DummyResponse("`done`")

    monkeypatch.setattr(llm_utils.genai, "Client", lambda api_key: DummyClient(generator=gen))

    resp = llm_utils.execute_llm_response(prompt="hi", max_retries=2, base_delay=0)
    assert resp is not None
    # normalized_text should be set and backticks removed
    assert getattr(resp, "normalized_text", "") == "done"
    assert calls["n"] == 2


def test_execute_llm_response_missing_api_key(monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    with pytest.raises(ValueError):
        llm_utils.execute_llm_response(prompt="hi")


def test_execute_llm_prompt_with_parts(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "dummy")

    seen = {}

    def gen(**kwargs):
        # The function under test should pass through list of parts when provided
        seen["contents"] = kwargs.get("contents")
        return DummyResponse("ok")

    monkeypatch.setattr(llm_utils.genai, "Client", lambda api_key: DummyClient(generator=gen))

    text = llm_utils.execute_llm_prompt(parts=["a", None, "b"])  # None should be filtered out upstream
    assert text == "ok"
    assert seen["contents"] == ["a", "b"]
