"""
Minimal Bayesian Change Point Model using PyMC3
Reuses existing Task 1 infrastructure
"""

import numpy as np
import pandas as pd
import pymc3 as pm
import matplotlib.pyplot as plt
from datetime import datetime
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from change_point_model import ChangePointModel
from event_compiler import EventCompiler

class BayesianChangePoint:
    def __init__(self, data_path):
        self.data = pd.read_csv(data_path)
        self.data['Date'] = pd.to_datetime(self.data['Date'], errors='coerce')
        self.data = self.data.dropna().sort_values('Date').reset_index(drop=True)
        self.log_returns = np.diff(np.log(self.data['Price'].values))
        self.trace = None
        
    def build_model(self):
        """Build Bayesian change point model"""
        n = len(self.log_returns)
        
        with pm.Model() as model:
            # Switch point
            tau = pm.DiscreteUniform('tau', lower=1, upper=n-2)
            
            # Parameters before/after change
            mu_1 = pm.Normal('mu_1', mu=0, sigma=0.1)
            mu_2 = pm.Normal('mu_2', mu=0, sigma=0.1)
            sigma = pm.HalfNormal('sigma', sigma=0.1)
            
            # Switch function
            idx = np.arange(n)
            mu = pm.math.switch(tau >= idx, mu_1, mu_2)
            
            # Likelihood
            obs = pm.Normal('obs', mu=mu, sigma=sigma, observed=self.log_returns)
            
        self.model = model
        return model
    
    def fit(self, draws=1000):
        """Fit the model"""
        with self.model:
            self.trace = pm.sample(draws=draws, tune=500, chains=2, 
                                 return_inferencedata=False, progressbar=False)
        return self.trace
    
    def get_change_point(self):
        """Get most probable change point"""
        tau_samples = self.trace['tau']
        tau_mode = int(np.round(np.mean(tau_samples)))
        change_date = self.data.iloc[tau_mode+1]['Date']  # +1 for log_returns offset
        return change_date, tau_mode
    
    def quantify_impact(self):
        """Quantify impact of change point"""
        change_date, tau_idx = self.get_change_point()
        
        before_prices = self.data.iloc[:tau_idx+1]['Price']
        after_prices = self.data.iloc[tau_idx+1:]['Price']
        
        before_mean = before_prices.mean()
        after_mean = after_prices.mean()
        change_pct = ((after_mean - before_mean) / before_mean) * 100
        
        return {
            'change_date': change_date,
            'before_mean': before_mean,
            'after_mean': after_mean,
            'change_percent': change_pct
        }

def associate_with_events(change_date, events_df, tolerance_days=30):
    """Associate change point with events"""
    change_date = pd.to_datetime(change_date)
    
    # Find events within tolerance
    time_diff = abs((events_df['date'] - change_date).dt.days)
    nearby_events = events_df[time_diff <= tolerance_days]
    
    if len(nearby_events) > 0:
        closest_event = nearby_events.loc[time_diff.idxmin()]
        days_diff = time_diff.min()
        return closest_event, days_diff
    
    return None, None

def main():
    """Main analysis pipeline"""
    # Load data
    data_path = "../../data/raw/brent_oil_prices.csv"
    
    # Initialize Bayesian model
    model = BayesianChangePoint(data_path)
    model.build_model()
    
    print("Fitting Bayesian change point model...")
    model.fit(draws=1000)
    
    # Get results
    impact = model.quantify_impact()
    change_date = impact['change_date']
    
    # Load events
    event_compiler = EventCompiler()
    events_df = event_compiler.compile_major_events()
    
    # Associate with events
    closest_event, days_diff = associate_with_events(change_date, events_df)
    
    # Print results
    print(f"\n=== BAYESIAN CHANGE POINT ANALYSIS ===")
    print(f"Detected change point: {change_date.strftime('%Y-%m-%d')}")
    print(f"Price change: ${impact['before_mean']:.2f} → ${impact['after_mean']:.2f}")
    print(f"Percentage change: {impact['change_percent']:.1f}%")
    
    if closest_event is not None:
        print(f"\nClosest event: {closest_event['event']}")
        print(f"Event date: {closest_event['date'].strftime('%Y-%m-%d')}")
        print(f"Days difference: {days_diff}")
        print(f"Event impact: {closest_event['impact']}")
        
        # Generate hypothesis
        direction = "before" if days_diff < 0 else "after"
        print(f"\nHypothesis: The change point is likely triggered by '{closest_event['event']}' "
              f"which occurred {days_diff} days {direction} the detected change, "
              f"resulting in a {impact['change_percent']:.1f}% change in mean price level.")
    else:
        print(f"\nNo major events found within 30 days of change point.")
    
    return model, impact, closest_event

if __name__ == "__main__":
    model, impact, event = main()