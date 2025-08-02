# Contributing to Brent Oil Change Point Analysis

We welcome contributions to improve this project! Here's how you can help:

## Getting Started

1. **Fork the repository** on GitHub
2. **Clone your fork** locally:
   ```bash
   git clone https://github.com/your-username/Change-point-analysis-and-statistical-modelling-of-time-series-data.git
   ```
3. **Create a new branch** for your feature:
   ```bash
   git checkout -b feature/your-feature-name
   ```

## Development Setup

1. **Install development dependencies**:
   ```bash
   pip install -r requirements.txt
   pip install pytest pytest-cov flake8 black
   ```

2. **Run tests** to ensure everything works:
   ```bash
   pytest tests/
   ```

## Making Changes

1. **Write your code** following the existing style
2. **Add tests** for new functionality in `tests/`
3. **Update documentation** if needed
4. **Run quality checks**:
   ```bash
   # Code formatting
   black src/ tests/
   
   # Linting
   flake8 src/ tests/
   
   # Tests
   pytest tests/ --cov=src
   ```

## Submitting Changes

1. **Commit your changes**:
   ```bash
   git add .
   git commit -m "Add: brief description of changes"
   ```

2. **Push to your fork**:
   ```bash
   git push origin feature/your-feature-name
   ```

3. **Open a Pull Request** on GitHub with:
   - Clear description of changes
   - Reference to any related issues
   - Screenshots if applicable

## Code Style

- Follow PEP 8 guidelines
- Use descriptive variable names
- Add docstrings to functions and classes
- Include type hints where appropriate

## Questions?

Open an issue for discussion before making major changes.