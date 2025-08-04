# Makefile for Change Point Analysis Project
# Provides automated workflows for setup, testing, and analysis

.PHONY: help install data analysis test lint clean all dashboard

# Default target
help:
	@echo "Available targets:"
	@echo "  install    - Install dependencies and setup environment"
	@echo "  data       - Process raw data and compile events"
	@echo "  analysis   - Run complete change point analysis"
	@echo "  dashboard  - Start web dashboard"
	@echo "  test       - Run test suite with coverage"
	@echo "  lint       - Check code style and quality"
	@echo "  clean      - Clean generated files"
	@echo "  all        - Run complete workflow (install + data + analysis + test)"

# Install dependencies
install:
	@echo "Installing dependencies..."
	pip install -r requirements.txt
	@echo "✓ Dependencies installed"

# Process data
data:
	@echo "Processing data and compiling events..."
	python -c "from src.event_compiler import EventCompiler; EventCompiler().compile_major_events()"
	@echo "✓ Data processing complete"

# Run analysis
analysis:
	@echo "Running change point analysis..."
	python changepoint_detection.py
	@echo "✓ Analysis complete"

# Start dashboard
dashboard:
	@echo "Starting web dashboard..."
	cd src/task3/backend && python app.py

# Run tests
test:
	@echo "Running test suite..."
	pytest tests/ --cov=src --cov-report=html --cov-report=term
	@echo "✓ Tests complete. Coverage report in htmlcov/"

# Code quality checks
lint:
	@echo "Checking code quality..."
	flake8 src/ --max-line-length=100 --ignore=E203,W503
	@echo "✓ Code quality check complete"

# Clean generated files
clean:
	@echo "Cleaning generated files..."
	rm -rf __pycache__ .pytest_cache htmlcov .coverage
	rm -rf src/__pycache__ src/*/__pycache__
	rm -f analysis.log
	@echo "✓ Cleanup complete"

# Complete workflow
all: install data analysis test
	@echo "✓ Complete workflow finished successfully"