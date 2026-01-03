.PHONY: test test-unit test-integration test-slow test-all test-coverage clean help

# Default test command - runs fast unit tests only
test: test-unit

# Run only fast unit tests (excludes slow and integration)
test-unit:
	uv run pytest tests/ -m "not slow and not integration" -v

# Run integration tests only
test-integration:
	uv run pytest tests/ -m "integration" -v

# Run slow tests only
test-slow:
	uv run pytest tests/ -m "slow" -v

# Run all tests (unit + integration + slow)
test-all:
	uv run pytest tests/ -v

# Run full test suite with coverage report
test-coverage:
	uv run pytest tests/ --cov=src/example_rnaseq --cov-report=term-missing --cov-report=html -v

# Clean up generated files
clean:
	rm -rf .pytest_cache
	rm -rf htmlcov
	rm -rf .coverage
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete

# Show help
help:
	@echo "Available targets:"
	@echo "  test            - Run fast unit tests (default)"
	@echo "  test-unit       - Run only fast unit tests (excludes slow/integration)"
	@echo "  test-integration- Run integration tests only"
	@echo "  test-slow       - Run slow tests only"
	@echo "  test-all        - Run all tests"
	@echo "  test-coverage   - Run all tests with coverage report"
	@echo "  clean           - Remove generated files"
	@echo "  help            - Show this help message"
