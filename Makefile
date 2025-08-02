# Makefile for Change Point Analysis Project

.PHONY: install clean data analysis test lint all

# Install dependencies
install:
	pip install -r requirements.txt

# Clean generated files
clean:
	rm -rf results/figures/*
	rm -rf results/models/*
	rm -f analysis.log
	rm -rf __pycache__/
	rm -rf .pytest_cache/

# Run data processing
data:
	python -c "from src.event_compiler import EventCompiler; EventCompiler().compile_major_events()"

# Run complete analysis
analysis:
	python changepoint_detection.py

# Run tests
test:
	pytest tests/ --cov=src

# Run linting
lint:
	flake8 src/ tests/
	black --check src/ tests/

# Run everything
all: install data analysis test