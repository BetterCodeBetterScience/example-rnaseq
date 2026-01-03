.PHONY: test test-unit test-integration test-slow test-all test-coverage test-data clean help

# Test data file path
TEST_DATA := tests/data/dataset-test_raw.h5ad

# Default test command - runs fast unit tests only
test: test-unit

# Run only fast unit tests (excludes slow and integration)
test-unit:
	uv run pytest tests/ -m "not slow and not integration" -v

# Create test data if it doesn't exist (requires DATADIR in .env and source data)
test-data: $(TEST_DATA)

$(TEST_DATA):
	@echo "Creating test dataset..."
	@echo "Note: Requires DATADIR to be set in .env file pointing to source data"
	uv run python tests/create_test_data.py

# Run integration tests only (requires test data)
test-integration: $(TEST_DATA)
	uv run pytest tests/ -m "integration" -v

# Run slow tests only (requires test data)
test-slow: $(TEST_DATA)
	uv run pytest tests/ -m "slow" -v

# Run all tests (unit + integration + slow, requires test data)
test-all: $(TEST_DATA)
	uv run pytest tests/ -v

# Run full test suite with coverage report (requires test data)
test-coverage: $(TEST_DATA)
	uv run pytest tests/ -W error::FutureWarning --cov=src/example_rnaseq --cov-report=term-missing --cov-report=html -v

# Clean up generated files
clean:
	rm -rf .pytest_cache
	rm -rf htmlcov
	rm -rf .coverage
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete

# Clean test data (to force regeneration)
clean-test-data:
	rm -f $(TEST_DATA)

# Show help
help:
	@echo "Available targets:"
	@echo "  test            - Run fast unit tests (default)"
	@echo "  test-unit       - Run only fast unit tests (excludes slow/integration)"
	@echo "  test-data       - Create test dataset if it doesn't exist"
	@echo "  test-integration- Run integration tests (creates test data if needed)"
	@echo "  test-slow       - Run slow tests (creates test data if needed)"
	@echo "  test-all        - Run all tests (creates test data if needed)"
	@echo "  test-coverage   - Run all tests with coverage (creates test data if needed)"
	@echo "  clean           - Remove generated files"
	@echo "  clean-test-data - Remove test data file (to force regeneration)"
	@echo "  help            - Show this help message"
	@echo ""
	@echo "Note: test-data, test-integration, test-slow, test-all, and test-coverage"
	@echo "      require DATADIR to be set in .env file pointing to the source data."
