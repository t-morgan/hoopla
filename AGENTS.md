# Agent Instructions for Hoopla Project

## ⚠️ CRITICAL: Package Management & Environment Setup

This project uses **uv** as its package manager. 

### 🚨 ALWAYS USE `uv run` FOR ALL COMMANDS

**Without `uv run`, you will get `ModuleNotFoundError` because the system Python doesn't have the dependencies installed.**

### 🔧 FIRST TIME SETUP (MUST DO THIS FIRST!)

Before running any tests or commands, the environment MUST be set up:

```bash
# Option 1: Use make (recommended)
make setup

# Option 2: Manual setup
uv sync                # Install dependencies
uv pip install -e .    # Install package in editable mode
```

**If you skip this step, tests will fail with ModuleNotFoundError!**

### Running Commands

✅ **CORRECT:**
```bash
uv run python main.py
uv run pytest                          # ← MUST use "uv run pytest", NOT just "pytest"
uv run pytest tests/test_actor_search.py
uv run pytest -v
uv run python -m cli.agentic_rag_cli
```

❌ **INCORRECT (WILL FAIL):**
```bash
python main.py          # ❌ Wrong Python
pytest                  # ❌ ModuleNotFoundError: No module named 'google.genai'
pip install -e .        # ❌ Wrong package manager
.venv/bin/pytest       # ❌ Direct venv access may fail
```

### Installing Dependencies

```bash
# Install project dependencies
uv sync

# Install specific package
uv pip install package-name

# Install project in editable mode
uv pip install -e .
```

### Testing

```bash
# Run all tests
uv run pytest

# Run only fast tests (for quick development feedback)
uv run pytest -m "not slow"

# Run only slow/integration tests
uv run pytest -m slow

# Run specific test file
uv run pytest tests/test_actor_search.py

# Run with verbose output
uv run pytest -v

# Using Makefile (recommended)
make test-fast     # Quick unit tests (~1-2s)
make test          # All tests
```

**Performance Note:** First test takes ~20s (loads ML models), but subsequent tests are fast due to session-scoped fixtures.

### Project Structure

- `cli/` - Main CLI commands and library code
- `cli/lib/` - Core library modules (agentic_rag, search utilities, etc.)
  - `cli/lib/agentic_tools/` - Modular search tool implementations (keyword, semantic, hybrid, regex, genre, actor)
- `data/` - Data files (movies.json, golden_dataset.json, etc.)
- `cache/` - Cached embeddings and indexes
- `tests/` - Test files
- `examples/` - Example scripts
- `docs/` - Documentation

### Important Notes

1. This project uses a flat layout with multiple top-level packages
2. The main package is `cli` with submodules in `cli/lib/`
3. `data/` and `cache/` are data directories, not Python packages
4. Always prefix Python commands with `uv run` to ensure proper environment activation

## 🔧 Troubleshooting

### "ModuleNotFoundError: No module named 'google.genai'"

**Problem:** You ran `pytest` instead of `uv run pytest`

**Solution:**
```bash
uv run pytest tests/test_actor_search.py
```

### "ImportError while importing test module"

**Problem:** Not using the project's virtual environment

**Solution:** Always use `uv run` before pytest or any Python command

### Setting up the Environment

If you get import errors even with `uv run`:

```bash
# Ensure virtual environment exists and dependencies are installed
uv sync

# Install development dependencies
uv pip install -e .

# Then run tests
uv run pytest
```

### Quick Reference

| Task | Command |
|------|---------|
| Run all tests | `uv run pytest` |
| Run specific test | `uv run pytest tests/test_actor_search.py` |
| Run with verbose | `uv run pytest -v` |
| Run specific test function | `uv run pytest tests/test_actor_search.py::test_actor_search_full_name_match_returns_results` |
| Install dependencies | `uv sync` |
| Install package editable | `uv pip install -e .` |

