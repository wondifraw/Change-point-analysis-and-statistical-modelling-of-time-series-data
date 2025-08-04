# Change Point Analysis and Statistical Modelling of Time Series Data

[![Python](https://img.shields.io/badge/python-3.8%2B-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![CI/CD](https://github.com/username/Change-point-analysis-and-statistical-modelling-of-time-series-data/workflows/CI%2FCD%20Pipeline/badge.svg)](https://github.com/username/Change-point-analysis-and-statistical-modelling-of-time-series-data/actions)
[![Code Coverage](https://img.shields.io/badge/coverage-85%25-green.svg)](https://github.com/username/Change-point-analysis-and-statistical-modelling-of-time-series-data)

A comprehensive analysis framework for detecting structural breaks in Brent oil price data using change point detection methods and statistical modeling techniques.

## Project Overview

This project implements a complete workflow for analyzing Brent oil prices to identify change points that correspond to major geopolitical events, OPEC decisions, and economic shocks. The analysis distinguishes between statistical correlation and causal relationships.

## Features

- **Modular Architecture**: Clean, maintainable code with comprehensive error handling
- **Multiple Change Point Methods**: PELT, Binary Segmentation, and Sliding Window detection
- **Bayesian Analysis**: Advanced Bayesian change point detection (Task 2)
- **Web Dashboard**: Interactive Flask-based dashboard with real-time visualizations
- **REST API**: Complete API endpoints for data access and integration
- **Event Analysis**: 15+ major oil market events with structured data
- **Time Series Analysis**: Trend, stationarity, and volatility analysis
- **Interactive Notebooks**: Ready-to-run Jupyter notebooks for analysis

## Project Structure

```
📦 Change-point-analysis-and-statistical-modelling-of-time-series-data/
├── 📁 .github/
│   └── 📁 workflows/              # CI/CD automation
│       ├── ci.yml                 # Testing pipeline
│       ├── deploy.yml             # Documentation deployment
│       └── release.yml            # Release automation
├── 📁 data/                       # Data storage (immutable raw, processed)
│   ├── 📁 raw/
│   │   └── brent_oil_prices.csv   # Original oil price data
│   └── 📁 processed/
│       └── events.csv             # Compiled geopolitical events
├── 📁 notebooks/                  # Jupyter notebooks for exploration
│   ├── 01_data_workflow_analysis.ipynb    # Complete workflow demo
│   ├── 02_events_analysis.ipynb           # Events analysis
│   └── 03_change_point_comparison.ipynb   # Methods comparison
├── 📁 src/                        # Source code modules
│   ├── 📁 task2/                 # Bayesian analysis components
│   │   ├── bayesian_model.py     # PyMC3-based Bayesian change point model
│   │   ├── bayesian_model_simple.py # Simplified Bayesian model (no PyMC3)
│   │   └── event_association.py  # Event-change point association
│   ├── 📁 task3/                 # Web application components
│   │   ├── 📁 backend/           # Flask API server
│   │   │   └── app.py            # Main Flask application
│   │   └── 📁 frontend/          # Web dashboard
│   │       └── dashboard.html    # Interactive HTML dashboard
│   ├── data_workflow.py          # Main workflow orchestrator
│   ├── event_compiler.py         # Geopolitical events compilation
│   ├── time_series_analyzer.py   # Time series properties analysis
│   └── change_point_model.py     # Change point detection models
├── 📁 scripts/                    # Automation scripts
│   └── run_analysis.py           # Automated analysis runner
├── 📁 tests/                      # Unit tests
│   └── test_workflow.py          # Test suite
├── 📁 results/                    # Generated files and results
│   ├── 📁 figures/               # Visualizations and plots
│   └── 📁 models/                # Saved model outputs
├── 📁 docs/                       # Project documentation
│   └── methodology.md            # Analysis methodology
├── 📄 changepoint_detection.py    # Main execution script
├── 📄 Makefile                   # Workflow automation
├── 📄 requirements.txt           # Python dependencies
├── 📄 environment.yml            # Conda environment
├── 📄 setup.py                   # Package installation
├── 📄 .gitignore                 # Git ignore rules
└── 📄 README.md                  # Project documentation
```

## Installation & Setup

**Requirements**: Python 3.8+ (Python 3.9 recommended)

### Option 1: Using pip
```bash
# Clone the repository
git clone <repository-url>
cd Change-point-analysis-and-statistical-modelling-of-time-series-data

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Option 2: Using conda
```bash
# Clone the repository
git clone <repository-url>
cd Change-point-analysis-and-statistical-modelling-of-time-series-data

# Create conda environment
conda env create -f environment.yml
conda activate brent-oil-analysis
```

### Option 3: Development installation
```bash
# For development with editable install
pip install -e .
```

## Step-by-Step Replication Guide

### Prerequisites
- Python 3.8+ installed
- Git installed
- 2GB free disk space

### Complete Setup (5 minutes)

1. **Clone and Navigate**
   ```bash
   git clone <repository-url>
   cd Change-point-analysis-and-statistical-modelling-of-time-series-data
   ```

2. **Create Environment**
   ```bash
   python -m venv venv
   # Windows:
   venv\Scripts\activate
   # macOS/Linux:
   source venv/bin/activate
   ```

3. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Verify Installation**
   ```bash
   python -c "import pandas, numpy, scipy; print('✓ Setup complete')"
   ```

5. **Run Analysis**
   ```bash
   python changepoint_detection.py
   ```

### Expected Results
- **Web Dashboard**: Interactive visualization at `http://127.0.0.1:5000/dashboard`
- **API Access**: REST endpoints available at `http://127.0.0.1:5000/api/*`
- **Console Output**: Analysis summary with detected change points
- **Generated Files**: 
  - `data/processed/events.csv` - Compiled events dataset
  - `analysis.log` - Detailed execution log
  - Results in `results/` directory

### Troubleshooting
- **Import errors**: Ensure virtual environment is activated
- **Permission errors**: Run with appropriate permissions
- **Missing data**: Check `data/raw/brent_oil_prices.csv` exists

## Usage

### 🌐 Web Dashboard (Recommended)
```bash
# Start the Flask web application
cd src/task3/backend
python app.py

# Open browser and navigate to:
# http://127.0.0.1:5000/dashboard
```
**Features**: Interactive charts, real-time data, change point visualization, event timeline

### 📊 API Access
```bash
# Health check
curl http://127.0.0.1:5000/api/health

# Get oil prices
curl "http://127.0.0.1:5000/api/oil-prices?start_date=2020-01-01"

# Get events
curl http://127.0.0.1:5000/api/events

# Get change points
curl http://127.0.0.1:5000/api/change-points
```

### 🔬 Complete Analysis Pipeline
```bash
# Run the complete analysis (command-line)
python changepoint_detection.py
```
**Expected output**: Analysis results in console, processed data in `data/processed/`, and log file `analysis.log`

### 📓 Interactive Analysis
```bash
# Launch Jupyter notebooks for step-by-step analysis
cd notebooks
jupyter notebook

# Start with: 01_data_workflow_analysis.ipynb
```

### 🧠 Bayesian Analysis (Task 2)
```bash
# Run Bayesian change point analysis
cd src/task2
python run_task2.py
```

### Testing
```bash
# Run all tests
pytest tests/

# Run tests with coverage
pytest tests/ --cov=src --cov-report=html
```

### Automated Workflows
```bash
# Using Makefile (recommended)
make install    # Install dependencies
make data       # Process data
make analysis   # Run complete analysis
make test       # Run tests
make lint       # Check code style
make all        # Run everything

# Using scripts
python scripts/run_analysis.py
```

## Sample Input and Output

### Input Data Format

**Brent Oil Prices (`data/raw/brent_oil_prices.csv`)**:
```csv
Date,Price
1987-05-20,18.63
1987-05-21,18.45
1987-05-22,18.55
...
2022-11-14,92.61
```

**Compiled Events (`data/processed/events.csv`)**:
```csv
date,event,category,impact
1990-08-02,Iraq invasion of Kuwait,Geopolitical,High
2001-09-11,September 11 attacks,Geopolitical,High
2008-09-15,Lehman Brothers collapse,Economic,High
...
```

### Expected Output

**Console Output (from `python changepoint_detection.py`)**:
```
=== ANALYSIS SUMMARY ===
Events compiled: 15
Change points detected: 3
Key assumptions: 4
Identified limitations: 5

=== TIME SERIES PROPERTIES ===
Trend Analysis: {'trend_slope': 0.023, 'trend_significance': True}
Stationarity Test: {'is_stationary': False, 'p_value': 0.342}

=== CHANGE POINT DETECTION ===
Method: PELT
Change points detected: 3
Change dates: ['2008-09-15', '2014-11-27', '2020-03-06']
```

**Generated Files**:
- `analysis.log` - Detailed execution log
- `data/processed/events.csv` - Compiled events dataset
- Notebook outputs with visualizations and analysis results

### Sample Visualization

**Events Timeline**:

![Events Timeline](results/figures/events_timeline.png)

**Brent Oil Prices Over Time**
![Brent Oil Prices](results/figures/brent_oil_prices_timeline.png)

*Visualization of major oil market events categorized by type (Geopolitical, Economic, OPEC Decisions) plotted over time to show the relationship between events and potential market disruptions.*

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

### 5. Bayesian Analysis (`src/task2/`)
- **bayesian_model.py**: PyMC3-based Bayesian change point detection
- **bayesian_model_simple.py**: Simplified version without PyMC3 dependency
- **event_association.py**: Associates detected change points with geopolitical events
- Quantifies uncertainty in change point locations

### 6. Web Application (`src/task3/`)
- **Flask Backend (`backend/app.py`)**:
  - RESTful API with 6 endpoints
  - Real-time data processing
  - CORS-enabled for frontend integration
  - Comprehensive error handling
- **Interactive Dashboard (`frontend/dashboard.html`)**:
  - Real-time price charts using Chart.js
  - Summary statistics display
  - Change point visualization
  - Event timeline with filtering

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

## File Dependencies

### Core Dependencies
```
src/data_workflow.py          → src/event_compiler.py
                             → src/time_series_analyzer.py  
                             → src/change_point_model.py

changepoint_detection.py     → src/data_workflow.py
                             → data/raw/brent_oil_prices.csv

notebooks/*.ipynb            → src/* (all modules)
                             → data/raw/brent_oil_prices.csv
```

### Data Flow
```
data/raw/brent_oil_prices.csv → src/event_compiler.py → data/processed/events.csv
                              → src/time_series_analyzer.py → analysis results
                              → src/change_point_model.py → change points
                              → results/figures/*.png
```

### Required Files for Execution
- `data/raw/brent_oil_prices.csv` (input data)
- `src/*.py` (all source modules)
- `requirements.txt` (dependencies)

### Generated Files
- `data/processed/events.csv` (compiled events)
- `analysis.log` (execution log)
- `results/figures/*.png` (visualizations)

## Dependencies

### Core Dependencies
- pandas >= 1.5.0
- numpy >= 1.21.0
- scipy >= 1.9.0
- statsmodels >= 0.13.0
- matplotlib >= 3.5.0
- seaborn >= 0.11.0
- jupyter >= 1.0.0

### Web Application
- flask >= 2.0.0
- flask-cors >= 3.0.0

### Bayesian Analysis (Optional)
- pymc3 >= 3.11.0 (for full Bayesian model)
- theano-pymc >= 1.1.0

### Development & Testing
- pytest >= 6.0.0
- nbval >= 0.9.6
- papermill >= 2.4.0

## API Endpoints

The Flask backend provides the following REST API endpoints:

| Endpoint | Method | Description |
|----------|--------|--------------|
| `/` | GET | API information and available endpoints |
| `/dashboard` | GET | Interactive web dashboard |
| `/api/health` | GET | System health check |
| `/api/oil-prices` | GET | Brent oil price data with optional date filtering |
| `/api/events` | GET | Geopolitical events with category/impact filtering |
| `/api/change-points` | GET | Detected change points with statistical measures |
| `/api/analysis-summary` | GET | Comprehensive analysis summary |
| `/api/event-impact-analysis` | GET | Event-price correlation analysis |

### Example API Usage
```bash
# Get recent oil prices
curl "http://127.0.0.1:5000/api/oil-prices?start_date=2022-01-01&end_date=2022-12-31"

# Filter high-impact geopolitical events
curl "http://127.0.0.1:5000/api/events?category=Geopolitical&impact=High"

# Get comprehensive analysis summary
curl "http://127.0.0.1:5000/api/analysis-summary"
```

## Communication Channels

Results can be communicated through:
1. **Interactive Web Dashboard** - Real-time visualization at `/dashboard`
2. **REST API** - Programmatic access to all analysis results
3. **Executive Summary** - Key metrics via `/api/analysis-summary`
4. **Technical Reports** - Detailed methodology in Jupyter notebooks
5. **Presentation Slides** - Stakeholder-ready visualizations

## Testing

Run the test suite to validate functionality:

```bash
# Run all tests
pytest tests/

# Run with coverage report
pytest tests/ --cov=src --cov-report=html

# View coverage report
open htmlcov/index.html  # On Windows: start htmlcov/index.html
```

**Test Coverage**: The project includes unit tests for core components:
- Event compilation and validation
- Time series analysis functions
- Change point detection methods
- Workflow orchestration

## Contributing

We welcome contributions! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines on:
- Setting up development environment
- Code style and standards
- Submitting pull requests
- Reporting issues

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

**Academic Use**: This project is designed for educational and research purposes in time series analysis and change point detection.

## Quick Start

1. **Clone and setup**:
   ```bash
   git clone <repository-url>
   cd Change-point-analysis-and-statistical-modelling-of-time-series-data
   pip install -r requirements.txt
   ```

2. **Launch web dashboard**:
   ```bash
   cd src/task3/backend
   python app.py
   ```

3. **Open browser**: `http://127.0.0.1:5000/dashboard`

4. **Explore the analysis**: Interactive charts, change points, and event correlations

## Architecture Overview

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Data Layer    │    │  Analysis Layer  │    │   Web Layer     │
│                 │    │                  │    │                 │
│ • Oil Prices    │───▶│ • Change Points  │───▶│ • Flask API     │
│ • Events Data   │    │ • Bayesian Model │    │ • Dashboard     │
│ • Processed     │    │ • Event Analysis │    │ • Visualizations│
└─────────────────┘    └──────────────────┘    └─────────────────┘
```