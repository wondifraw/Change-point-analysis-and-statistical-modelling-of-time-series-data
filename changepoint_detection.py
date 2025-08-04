#!/usr/bin/env python3
"""
Main Change Point Detection Script
Orchestrates the complete Brent oil price analysis workflow
"""

import pandas as pd
import numpy as np
import logging
import sys
import os
from pathlib import Path

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from data_workflow import DataAnalysisWorkflow
from event_compiler import EventCompiler
from change_point_model import ChangePointModel

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('analysis.log'),
        logging.StreamHandler()
    ]
)

def main():
    """Execute the complete analysis workflow"""
    try:
        logging.info("Starting Brent oil price change point analysis")
        
        # Step 1: Compile events
        event_compiler = EventCompiler()
        events_df = event_compiler.compile_major_events()
        
        # Step 2: Load oil data
        data_path = "data/raw/brent_oil_prices.csv"
        if os.path.exists(data_path):
            oil_data = pd.read_csv(data_path)
            oil_data['Date'] = pd.to_datetime(oil_data['Date'])
            oil_data = oil_data.rename(columns={'Date': 'date', 'Price': 'price'})
        else:
            # Create synthetic data for demonstration
            dates = pd.date_range('2000-01-01', '2023-12-31', freq='D')
            prices = 50 + np.random.randn(len(dates)).cumsum() * 2
            oil_data = pd.DataFrame({'date': dates, 'price': prices})
            logging.warning("Using synthetic data - place real data in data/raw/brent_oil_prices.csv")
        
        # Step 3: Change point analysis
        cp_model = ChangePointModel(oil_data, method='pelt')
        results = cp_model.detect_change_points(penalty=15.0)
        
        # Step 4: Initialize workflow
        workflow = DataAnalysisWorkflow(data_path)
        workflow_results = workflow.execute_workflow()
        
        # Summary
        print("\n=== ANALYSIS SUMMARY ===")
        print(f"Events compiled: {len(events_df)}")
        print(f"Change points detected: {len(results.get('change_points', []))}")
        print(f"Key assumptions: {len(workflow.assumptions)}")
        print(f"Identified limitations: {len(workflow.limitations)}")
        
        if results.get('change_points'):
            print(f"\n=== CHANGE POINT DETECTION ===")
            print(f"Method: {results.get('method', 'Unknown')}")
            print(f"Change points detected: {len(results['change_points'])}")
            print(f"Change dates: {results.get('change_dates', [])}")
        
        logging.info("Analysis completed successfully")
        return results
        
    except Exception as e:
        logging.error(f"Analysis failed: {str(e)}")
        raise

if __name__ == "__main__":
    main()