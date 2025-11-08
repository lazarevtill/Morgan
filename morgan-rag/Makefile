.PHONY: format format-check lint test

# Format code using black
format:
	black morgan/core/ morgan/services/ tests/
	@echo "Code formatting complete!"

# Check formatting without making changes
format-check:
	black --check morgan/core/ morgan/services/ tests/

# Run linting (if you add other tools)
lint: format-check
	@echo "Linting complete!"

# Run tests
test:
	pytest tests/

