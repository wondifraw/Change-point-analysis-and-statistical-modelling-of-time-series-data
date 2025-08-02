# Contributing

## Branch Naming Strategy

- `feature/description` - New features
- `fix/description` - Bug fixes
- `docs/description` - Documentation updates
- `test/description` - Test improvements

Example: `feature/add-arima-model` or `fix/memory-leak`

## How to Submit PRs

1. Fork the repository
2. Create a branch: `git checkout -b feature/your-feature`
3. Make changes and commit: `git commit -m "Add: brief description"`
4. Push: `git push origin feature/your-feature`
5. Open a Pull Request with:
   - Clear title and description
   - Reference related issues (#123)
   - List of changes made

## Code Style Guidelines

- **Follow PEP 8** for Python code formatting
- **Use descriptive names** for variables and functions
- **Add docstrings** to all functions and classes
- **Include type hints** where possible
- **Keep functions small** and focused on single tasks

```python
def analyze_trend(data: pd.DataFrame) -> Dict[str, float]:
    """Analyze trend in time series data.
    
    Args:
        data: DataFrame with 'date' and 'price' columns
        
    Returns:
        Dictionary with trend analysis results
    """
```

## Testing Instructions

### Run Tests
```bash
# All tests
pytest tests/

# With coverage
pytest tests/ --cov=src --cov-report=html

# Specific test file
pytest tests/test_workflow.py
```

### Add Tests
- Create test files in `tests/` directory
- Name test functions with `test_` prefix
- Test both success and failure cases
- Aim for >80% code coverage

### Code Quality Checks
```bash
# Linting
flake8 src/ tests/

# Type checking (optional)
mypy src/
```

## Questions?

Open an issue for discussion before major changes.