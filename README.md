# Change Point Analysis and Statistical Modelling of Time Series Data

A comprehensive analysis framework for detecting structural breaks in Brent oil price data using change point detection methods and statistical modeling techniques.

## Project Overview

This project implements a complete workflow for analyzing Brent oil prices to identify change points that correspond to major geopolitical events, OPEC decisions, and economic shocks. The analysis distinguishes between statistical correlation and causal relationships.

## Features

- **Modular Architecture**: Clean, maintainable code with comprehensive error handling
- **Multiple Change Point Methods**: PELT, Binary Segmentation, and Sliding Window detection
- **Event Analysis**: 15+ major oil market events with structured data
- **Time Series Analysis**: Trend, stationarity, and volatility analysis
- **Interactive Notebooks**: Ready-to-run Jupyter notebooks for analysis

## Project Structure

```
├── src/
│   ├── data_workflow.py          # Main workflow orchestrator
│   ├── event_compiler.py         # Geopolitical events compilation
│   ├── time_series_analyzer.py   # Time series properties analysis
│   └── change_point_model.py     # Change point detection models
├── notebooks/
│   ├── 01_data_workflow_analysis.ipynb    # Complete workflow demo
│   ├── 02_events_analysis.ipynb           # Events analysis
│   └── 03_change_point_comparison.ipynb   # Methods comparison
├── data/
│   ├── raw/
│   │   └── BrentOilPrices.csv
│   └── processed/
│       └── events.csv
├── main.py                       # Main execution script
└── requirements.txt              # Dependencies
```

## Installation

1. Clone the repository
2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Quick Start
```python
python main.py
```

### Jupyter Notebooks
```bash
cd notebooks
jupyter notebook 01_data_workflow_analysis.ipynb
```

## Key Components

### 1. Data Workflow (`src/data_workflow.py`)
- Orchestrates complete analysis pipeline
- Defines assumptions and limitations
- Handles correlation vs causation distinction

### 2. Event Compiler (`src/event_compiler.py`)
- Compiles 15+ major oil market events
- Categories: Geopolitical, Economic, OPEC Decisions
- Structured dataset with dates and impact levels

### 3. Time Series Analyzer (`src/time_series_analyzer.py`)
- Trend analysis using linear regression
- Stationarity testing (ADF test)
- Volatility clustering detection
- Modeling implications derivation

### 4. Change Point Model (`src/change_point_model.py`)
- PELT (Pruned Exact Linear Time) detection
- Binary Segmentation method
- Sliding Window approach
- Expected outputs and limitations documentation

## Major Events Analyzed

- Iraq invasion of Kuwait (1990)
- September 11 attacks (2001)
- Iraq War begins (2003)
- Lehman Brothers collapse (2008)
- Arab Spring begins (2010)
- Libya civil war intervention (2011)
- OPEC production decisions (2014-2023)
- COVID-19 pandemic (2020)
- Russia-Ukraine conflict (2022)

## Key Assumptions

1. Oil price data is accurate and complete
2. Change points represent structural breaks in market behavior
3. Geopolitical events have measurable impact on oil prices
4. Time series exhibits non-stationary behavior with potential regime changes

## Critical Limitations

1. **Statistical correlation does not imply causal relationship**
2. External factors beyond modeled events may influence prices
3. Change point detection may identify spurious breaks
4. Model assumes independence of residuals which may not hold
5. Limited to historical data patterns, may not predict future behavior

## Correlation vs Causation

This analysis identifies **statistical correlations** between events and price changes but **cannot establish causation**. Change point detection shows when statistical properties change, but proving that specific events caused these changes requires:

- Ruling out confounding factors
- Theoretical framework
- Additional empirical evidence
- Domain expertise for interpretation

## Dependencies

- pandas >= 1.5.0
- numpy >= 1.21.0
- scipy >= 1.9.0
- statsmodels >= 0.13.0
- matplotlib >= 3.5.0
- seaborn >= 0.11.0

## Communication Channels

Results can be communicated through:
1. Executive dashboard with key metrics
2. Technical report with detailed methodology
3. Interactive visualizations for exploratory analysis
4. Presentation slides for stakeholder meetings

## License

This project is for educational and research purposes.