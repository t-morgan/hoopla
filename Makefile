.PHONY: test test-v test-fast test-slow test-unit test-integration test-actor test-llm install sync clean help setup verify

# Default target
help:
	@echo "Hoopla Project - Common Commands"
	@echo "================================="
	@echo ""
	@echo "Setup:"
	@echo "  make setup          - Complete setup (sync + install + verify)"
	@echo "  make sync           - Sync dependencies with uv"
	@echo "  make install        - Install package in editable mode"
	@echo "  make verify         - Verify environment is working"
	@echo ""
	@echo "Testing:"
	@echo "  make test           - Run all tests"
	@echo "  make test-v         - Run all tests with verbose output"
	@echo "  make test-fast      - Run only fast unit tests (~1-2s)"
	@echo "  make test-slow      - Run only slow integration tests (~20s)"
	@echo "  make test-unit      - Run only unit tests (mocked, fast)"
	@echo "  make test-integration - Run only integration tests (real deps)"
	@echo "  make test-actor     - Run actor search tests"
	@echo "  make test-llm       - Run LLM utility tests"
	@echo ""
	@echo "Cleanup:"
	@echo "  make clean          - Remove cache files and build artifacts"
	@echo ""
	@echo "⚠️  All commands use 'uv run' automatically"
	@echo "💡 Tip: Use 'make test-fast' for quick feedback during development"

# Setup commands
sync:
	@echo "📦 Syncing dependencies..."
	uv sync

install: sync
	@echo "📦 Installing package in editable mode..."
	uv pip install -e .

verify:
	@echo "🔍 Verifying environment..."
	@echo "Checking pytest..."
	@uv run pytest --version || (echo "❌ pytest not available" && exit 1)
	@echo "Checking google.genai..."
	@uv run python -c "import google.genai; print('✓ google.genai available')" || (echo "❌ google.genai not available" && exit 1)
	@echo "Checking cli package..."
	@uv run python -c "from cli.lib.agentic_rag import ActorSearchTool; print('✓ cli.lib.agentic_rag available')" || (echo "❌ cli package not available" && exit 1)
	@echo "✅ Environment is properly configured!"

setup: sync install verify
	@echo "✅ Setup complete! Run 'make test' to run tests."


# Test commands
test:
	uv run pytest

test-v:
	uv run pytest -v

test-fast:
	@echo "🚀 Running fast tests only (unit tests)..."
	uv run pytest -m "not slow"

test-slow:
	@echo "🐢 Running slow tests only (integration tests)..."
	uv run pytest -m slow

test-unit:
	@echo "⚡ Running unit tests..."
	uv run pytest -m unit

test-integration:
	@echo "🔗 Running integration tests..."
	uv run pytest -m integration

test-actor:
	uv run pytest tests/test_actor_search.py -v

test-llm:
	uv run pytest tests/test_llm_utils.py -v

# Cleanup
clean:
	rm -rf __pycache__
	rm -rf .pytest_cache
	rm -rf *.egg-info
	rm -rf build/
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete

