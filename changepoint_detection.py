"""
Main Analysis Script
===================

Orchestrates the complete Brent oil price change point analysis workflow.
"""

import pandas as pd
import logging
from pathlib import Path
from src.data_workflow import DataAnalysisWorkflow
from src.event_compiler import EventCompiler
from src.time_series_analyzer import TimeSeriesAnalyzer
from src.change_point_model import ChangePointModel

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
    """Execute the complete analysis workflow."""
    try:
        # Initialize paths
        data_path = "data/raw/brent_oil_prices.csv"
        events_path = "data/processed/events.csv"
        
        logging.info("Starting Brent oil price change point analysis")
        
        # Step 1: Initialize workflow
        workflow = DataAnalysisWorkflow(data_path, events_path)
        
        # Step 2: Compile events data
        event_compiler = EventCompiler(events_path)
        events_df = event_compiler.compile_major_events()
        validation = event_compiler.validate_events_data(events_df)
        
        if not validation.get('has_minimum_events', False):
            raise ValueError("Insufficient events data compiled")
        
        # Step 3: Load and analyze time series data
        try:
            oil_data = pd.read_csv(data_path)
            oil_data['Date'] = pd.to_datetime(oil_data['Date'])
            oil_data = oil_data.rename(columns={'Date': 'date', 'Price': 'price'})
            
            # Analyze time series properties
            ts_analyzer = TimeSeriesAnalyzer(oil_data)
            properties = ts_analyzer.analyze_properties()
            
            logging.info("Time series properties analysis completed")
            
        except FileNotFoundError:
            logging.warning("Oil price data not found, using synthetic data for demonstration")
            # Create synthetic data for demonstration
            dates = pd.date_range('2000-01-01', '2023-12-31', freq='D')
            prices = 50 + np.random.randn(len(dates)).cumsum() * 2
            oil_data = pd.DataFrame({'date': dates, 'price': prices})
            
            ts_analyzer = TimeSeriesAnalyzer(oil_data)
            properties = ts_analyzer.analyze_properties()
        
        # Step 4: Change point analysis
        cp_model = ChangePointModel(oil_data, method='pelt')
        change_points = cp_model.detect_change_points(penalty=15.0)
        expected_outputs = cp_model.get_expected_outputs()
        
        # Step 5: Execute complete workflow
        workflow_results = workflow.execute_workflow()
        
        # Step 6: Summary results
        results_summary = {
            'workflow_status': 'Completed',
            'events_compiled': len(events_df),
            'time_series_properties': properties,
            'change_points_detected': len(change_points.get('change_points', [])),
            'assumptions': workflow.assumptions,
            'limitations': workflow.limitations,
            'expected_outputs': expected_outputs
        }
        
        logging.info("Analysis workflow completed successfully")
        print("\n=== ANALYSIS SUMMARY ===")
        print(f"Events compiled: {results_summary['events_compiled']}")
        print(f"Change points detected: {results_summary['change_points_detected']}")
        print(f"Key assumptions: {len(results_summary['assumptions'])}")
        print(f"Identified limitations: {len(results_summary['limitations'])}")
        
        return results_summary
        
    except Exception as e:
        logging.error(f"Analysis workflow failed: {str(e)}")
        raise

if __name__ == "__main__":
    import numpy as np
    results = main()