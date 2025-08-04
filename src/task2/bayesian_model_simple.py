"""
Simplified Bayesian Change Point Model (without PyMC3 dependency)
Provides basic functionality for the Flask app
"""

import numpy as np
import pandas as pd
from datetime import datetime
import sys
import os

class BayesianChangePoint:
    def __init__(self, data_path=None, data=None):
        if data is not None:
            self.data = data
        else:
            self.data = pd.read_csv(data_path)
            self.data['Date'] = pd.to_datetime(self.data['Date'], errors='coerce')
            self.data = self.data.dropna().sort_values('Date').reset_index(drop=True)
        
        self.log_returns = np.diff(np.log(self.data['Price'].values))
        self.change_points = []
        
    def detect_change_points(self, method='simple'):
        """Simple change point detection without PyMC3"""
        prices = self.data['Price'].values
        n = len(prices)
        
        # Simple variance-based change point detection
        best_split = None
        best_score = float('inf')
        
        for i in range(20, n-20):  # Avoid edges
            before = prices[:i]
            after = prices[i:]
            
            # Calculate combined variance
            var_before = np.var(before)
            var_after = np.var(after)
            combined_var = (len(before) * var_before + len(after) * var_after) / n
            
            if combined_var < best_score:
                best_score = combined_var
                best_split = i
        
        if best_split:
            self.change_points = [best_split]
            return [self.data.iloc[best_split]['Date']]
        
        return []
    
    def quantify_impact(self):
        """Quantify impact of detected change points"""
        if not self.change_points:
            self.detect_change_points()
        
        if not self.change_points:
            return None
        
        tau_idx = self.change_points[0]
        change_date = self.data.iloc[tau_idx]['Date']
        
        before_prices = self.data.iloc[:tau_idx]['Price']
        after_prices = self.data.iloc[tau_idx:]['Price']
        
        before_mean = before_prices.mean()
        after_mean = after_prices.mean()
        change_pct = ((after_mean - before_mean) / before_mean) * 100
        
        return {
            'change_date': change_date,
            'before_mean': float(before_mean),
            'after_mean': float(after_mean),
            'change_percent': float(change_pct),
            'method': 'Simple Variance-based'
        }

def associate_with_events(change_date, events_df, tolerance_days=30):
    """Associate change point with events"""
    if events_df is None or len(events_df) == 0:
        return None, None
        
    change_date = pd.to_datetime(change_date)
    
    # Find events within tolerance
    time_diff = abs((events_df['date'] - change_date).dt.days)
    nearby_events = events_df[time_diff <= tolerance_days]
    
    if len(nearby_events) > 0:
        closest_event = nearby_events.loc[time_diff.idxmin()]
        days_diff = int(time_diff.min())
        return closest_event, days_diff
    
    return None, None