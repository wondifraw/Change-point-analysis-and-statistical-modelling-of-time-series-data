"""
Task 2 Main Runner - Bayesian Change Point Analysis
Reuses Task 1 infrastructure with minimal additions
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import pandas as pd
import numpy as np
from datetime import datetime
from bayesian_model import BayesianChangePoint, associate_with_events
from data_workflow import DataWorkflow

def run_task2_analysis():
    """Run complete Task 2 analysis pipeline"""
    
    print("=" * 60)
    print("TASK 2: BAYESIAN CHANGE POINT ANALYSIS")
    print("=" * 60)
    
    # Initialize workflow (reuse Task 1)
    workflow = DataWorkflow()
    
    # Load and prepare data
    print("\n1. Loading and preparing data...")
    oil_data = pd.read_csv("data/raw/brent_oil_prices.csv")
    
    # Handle date parsing
    try:
        oil_data['Date'] = pd.to_datetime(oil_data['Date'], format='%d-%b-%y')
    except:
        try:
            oil_data['Date'] = pd.to_datetime(oil_data['Date'], format='%b %d, %Y')
        except:
            oil_data['Date'] = pd.to_datetime(oil_data['Date'])
    
    oil_data = oil_data.sort_values('Date').reset_index(drop=True)
    print(f"   Data loaded: {len(oil_data)} observations from {oil_data['Date'].min()} to {oil_data['Date'].max()}")
    
    # Initialize Bayesian model
    print("\n2. Building Bayesian change point model...")
    model = BayesianChangePoint("data/raw/brent_oil_prices.csv")
    model.build_model()
    
    # Fit model
    print("   Fitting model with MCMC sampling...")
    model.fit(draws=1000)
    print("   Model fitted successfully")
    
    # Get change point and impact
    print("\n3. Analyzing change point...")
    impact = model.quantify_impact()
    change_date = impact['change_date']
    
    # Load events (reuse Task 1)
    print("\n4. Loading geopolitical events...")
    events_df = workflow.event_compiler.compile_major_events()
    print(f"   Loaded {len(events_df)} major events")
    
    # Associate with events
    print("\n5. Associating change point with events...")
    closest_event, days_diff = associate_with_events(change_date, events_df)
    
    # Generate results
    print("\n" + "=" * 60)
    print("RESULTS SUMMARY")
    print("=" * 60)
    
    print(f"\nMost Probable Change Point: {change_date.strftime('%Y-%m-%d')}")
    print(f"Price Impact:")
    print(f"  - Before change: ${impact['before_mean']:.2f}")
    print(f"  - After change:  ${impact['after_mean']:.2f}")
    print(f"  - Change amount: ${impact['after_mean'] - impact['before_mean']:.2f}")
    print(f"  - Percentage change: {impact['change_percent']:.1f}%")
    
    if closest_event is not None:
        print(f"\nAssociated Event:")
        print(f"  - Event: {closest_event['event']}")
        print(f"  - Date: {closest_event['date'].strftime('%Y-%m-%d')}")
        print(f"  - Category: {closest_event['category']}")
        print(f"  - Impact Level: {closest_event['impact']}")
        print(f"  - Days from change point: {days_diff}")
        
        # Generate hypothesis
        direction = "before" if (closest_event['date'] - change_date).days < 0 else "after"
        print(f"\nHypothesis:")
        print(f"Following the {closest_event['event']} around {closest_event['date'].strftime('%Y-%m-%d')}, "
              f"the model detects a change point, with the average daily price shifting "
              f"from ${impact['before_mean']:.2f} to ${impact['after_mean']:.2f}, "
              f"an {'increase' if impact['change_percent'] > 0 else 'decrease'} of {abs(impact['change_percent']):.1f}%.")
    else:
        print(f"\nNo major events found within 30 days of the change point.")
        print("This change may be due to:")
        print("  - Cumulative effects of multiple smaller events")
        print("  - Market sentiment changes")
        print("  - Technical trading factors")
        print("  - Unidentified external factors")
    
    # Key assumptions and limitations
    print(f"\n" + "=" * 60)
    print("KEY ASSUMPTIONS & LIMITATIONS")
    print("=" * 60)
    
    assumptions = workflow.get_assumptions()
    limitations = workflow.get_limitations()
    
    print("\nKey Assumptions:")
    for i, assumption in enumerate(assumptions, 1):
        print(f"  {i}. {assumption}")
    
    print("\nCritical Limitations:")
    for i, limitation in enumerate(limitations, 1):
        print(f"  {i}. {limitation}")
    
    print(f"\n" + "=" * 60)
    print("CORRELATION vs CAUSATION")
    print("=" * 60)
    print("IMPORTANT: This analysis identifies statistical correlations between")
    print("events and price changes but CANNOT establish causation. Change point")
    print("detection shows when statistical properties change, but proving that")
    print("specific events caused these changes requires additional evidence.")
    
    return {
        'change_date': change_date,
        'impact': impact,
        'closest_event': closest_event,
        'days_diff': days_diff,
        'model': model
    }

if __name__ == "__main__":
    results = run_task2_analysis()