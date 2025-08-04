#!/usr/bin/env python3
"""
Task 2 Runner - Bayesian Change Point Analysis
Executes Bayesian inference for change point detection
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

try:
    from bayesian_model import BayesianChangePoint
    print("✓ Using full Bayesian model with PyMC3")
except ImportError:
    from bayesian_model_simple import BayesianChangePoint
    print("✓ Using simplified Bayesian model")

from bayesian_inference import BayesianChangePointMCMC
import pandas as pd
import numpy as np

def main():
    """Run Bayesian change point analysis"""
    print("🧠 Starting Bayesian Change Point Analysis")
    
    # Load data
    data_path = "../../data/raw/brent_oil_prices.csv"
    if os.path.exists(data_path):
        data = pd.read_csv(data_path)
        prices = data['Price'].values
    else:
        # Synthetic data
        np.random.seed(42)
        prices = np.concatenate([
            np.random.normal(50, 5, 100),
            np.random.normal(70, 8, 100),
            np.random.normal(60, 6, 100)
        ])
        print("⚠️ Using synthetic data")
    
    # Run MCMC analysis
    mcmc_model = BayesianChangePointMCMC(prices)
    results = mcmc_model.mcmc_sample(n_samples=1000, burn_in=200)
    
    print(f"\n📊 MCMC Results:")
    print(f"Acceptance rate: {results['acceptance_rate']:.2%}")
    print(f"Most probable # change points: {results['posterior_stats']['most_probable_n_changepoints']}")
    
    # Get point estimates
    estimates = mcmc_model.get_point_estimates()
    if estimates['most_probable_changepoints']:
        print(f"Change points: {estimates['most_probable_changepoints']}")
    
    return results

if __name__ == "__main__":
    main()