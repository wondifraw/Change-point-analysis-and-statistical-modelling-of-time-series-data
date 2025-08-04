# Task 2: Bayesian Change Point Analysis

## Overview
Implements Bayesian change point detection using PyMC3, reusing Task 1 infrastructure.

## Core Implementation
- **bayesian_model.py**: Minimal PyMC3 change point model
- **run_task2.py**: Main analysis pipeline reusing Task 1 components

## Key Features
1. **Bayesian Change Point Detection**: Uses PyMC3 to identify structural breaks
2. **Event Association**: Links change points to geopolitical events from Task 1
3. **Impact Quantification**: Measures price changes before/after change points
4. **Hypothesis Generation**: Formulates event-change relationships

## Usage
```python
from task2.run_task2 import run_task2_analysis
results = run_task2_analysis()
```

## Dependencies
Install PyMC3 requirements:
```bash
pip install -r task2/requirements.txt
```

## Output
- Change point dates with confidence intervals
- Quantified price impacts ($ and %)
- Event associations and hypotheses
- Correlation vs causation warnings