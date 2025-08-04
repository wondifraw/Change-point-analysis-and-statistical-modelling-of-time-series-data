"""
Event Association Module for Change Point Analysis

This module associates detected change points with geopolitical and economic events
to formulate hypotheses about which events likely triggered the detected shifts.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import seaborn as sns

class EventAssociator:
    """
    Associates detected change points with historical events.
    """
    
    def __init__(self, events_file_path):
        """
        Initialize with events data.
        
        Parameters:
        -----------
        events_file_path : str
            Path to the CSV file containing events data
        """
        self.events_df = pd.read_csv(events_file_path)
        self.events_df['date'] = pd.to_datetime(self.events_df['date'])
        
    def find_associated_events(self, change_points, tolerance_days=30):
        """
        Find events associated with detected change points.
        
        Parameters:
        -----------
        change_points : list
            List of change point dates (datetime objects)
        tolerance_days : int
            Number of days before/after change point to consider for association
            
        Returns:
        --------
        list : List of dictionaries containing change point and associated events
        """
        associations = []
        
        for cp_date in change_points:
            # Convert to datetime if string
            if isinstance(cp_date, str):
                cp_date = pd.to_datetime(cp_date)
            
            # Find events within tolerance window
            start_date = cp_date - timedelta(days=tolerance_days)
            end_date = cp_date + timedelta(days=tolerance_days)
            
            nearby_events = self.events_df[
                (self.events_df['date'] >= start_date) & 
                (self.events_df['date'] <= end_date)
            ].copy()
            
            # Calculate days difference
            nearby_events['days_from_cp'] = (nearby_events['date'] - cp_date).dt.days
            
            # Sort by proximity to change point
            nearby_events = nearby_events.sort_values('days_from_cp', key=abs)
            
            association = {
                'change_point_date': cp_date,
                'associated_events': nearby_events.to_dict('records'),
                'closest_event': nearby_events.iloc[0].to_dict() if len(nearby_events) > 0 else None,
                'num_events_nearby': len(nearby_events)
            }
            
            associations.append(association)
        
        return associations
    
    def generate_hypotheses(self, associations, impact_data=None):
        """
        Generate hypotheses about event-change point relationships.
        
        Parameters:
        -----------
        associations : list
            Output from find_associated_events()
        impact_data : dict, optional
            Impact quantification data from change point analysis
            
        Returns:
        --------
        list : List of hypothesis dictionaries
        """
        hypotheses = []
        
        for assoc in associations:
            cp_date = assoc['change_point_date']
            closest_event = assoc['closest_event']
            
            if closest_event is None:
                hypothesis = {
                    'change_point_date': cp_date,
                    'hypothesis': f"Change point on {cp_date.strftime('%Y-%m-%d')} may be due to unidentified market factors or cumulative effects of multiple smaller events.",
                    'confidence': 'Low',
                    'supporting_events': [],
                    'event_type': 'Unknown'
                }
            else:
                days_diff = abs(closest_event['days_from_cp'])
                event_name = closest_event['event']
                event_category = closest_event['category']
                event_impact = closest_event['impact']
                
                # Determine confidence based on proximity and event impact
                if days_diff <= 7 and event_impact == 'High':
                    confidence = 'High'
                elif days_diff <= 14 and event_impact in ['High', 'Medium']:
                    confidence = 'Medium'
                else:
                    confidence = 'Low'
                
                # Generate hypothesis text
                direction = "before" if closest_event['days_from_cp'] < 0 else "after"
                
                hypothesis_text = f"Change point on {cp_date.strftime('%Y-%m-%d')} is likely triggered by '{event_name}' ({event_category}) which occurred {days_diff} days {direction} the detected change."
                
                if impact_data:
                    hypothesis_text += f" This resulted in a {impact_data.get('percent_change', 0):.2f}% change in mean price level."
                
                hypothesis = {
                    'change_point_date': cp_date,
                    'hypothesis': hypothesis_text,
                    'confidence': confidence,
                    'supporting_events': assoc['associated_events'],
                    'closest_event': closest_event,
                    'event_type': event_category,
                    'days_difference': days_diff
                }
            
            hypotheses.append(hypothesis)
        
        return hypotheses
    
    def create_timeline_plot(self, change_points, figsize=(15, 8)):
        """
        Create a timeline plot showing change points and events.
        
        Parameters:
        -----------
        change_points : list
            List of change point dates
        figsize : tuple
            Figure size
            
        Returns:
        --------
        matplotlib.figure.Figure : The created figure
        """
        fig, ax = plt.subplots(figsize=figsize)
        
        # Plot events
        for idx, row in self.events_df.iterrows():
            y_pos = {'Geopolitical': 1, 'Economic': 2, 'OPEC Decision': 3}.get(row['category'], 1)
            color = {'High': 'red', 'Medium': 'orange', 'Low': 'yellow'}.get(row['impact'], 'gray')
            
            ax.scatter(row['date'], y_pos, c=color, s=100, alpha=0.7, edgecolors='black')
            ax.annotate(row['event'], (row['date'], y_pos), 
                       xytext=(5, 5), textcoords='offset points', 
                       fontsize=8, rotation=45, ha='left')
        
        # Plot change points
        for cp_date in change_points:
            if isinstance(cp_date, str):
                cp_date = pd.to_datetime(cp_date)
            ax.axvline(cp_date, color='blue', linestyle='--', linewidth=2, alpha=0.8)
            ax.text(cp_date, 3.5, f'Change Point\n{cp_date.strftime("%Y-%m-%d")}', 
                   rotation=90, ha='center', va='bottom', fontsize=10, color='blue')
        
        # Formatting
        ax.set_yticks([1, 2, 3])
        ax.set_yticklabels(['Geopolitical', 'Economic', 'OPEC Decision'])
        ax.set_xlabel('Date')
        ax.set_ylabel('Event Category')
        ax.set_title('Timeline of Events and Detected Change Points')
        ax.grid(True, alpha=0.3)
        
        # Legend
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor='red', label='High Impact'),
            Patch(facecolor='orange', label='Medium Impact'),
            Patch(facecolor='yellow', label='Low Impact'),
            plt.Line2D([0], [0], color='blue', linestyle='--', label='Change Point')
        ]
        ax.legend(handles=legend_elements, loc='upper left')
        
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        return fig
    
    def create_summary_report(self, hypotheses, output_file=None):
        """
        Create a summary report of the event-change point associations.
        
        Parameters:
        -----------
        hypotheses : list
            List of hypothesis dictionaries
        output_file : str, optional
            Path to save the report
            
        Returns:
        --------
        str : The report text
        """
        report = "CHANGE POINT - EVENT ASSOCIATION REPORT\n"
        report += "=" * 50 + "\n\n"
        
        report += f"Analysis Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
        report += f"Number of Change Points Analyzed: {len(hypotheses)}\n\n"
        
        # Summary statistics
        high_confidence = sum(1 for h in hypotheses if h['confidence'] == 'High')
        medium_confidence = sum(1 for h in hypotheses if h['confidence'] == 'Medium')
        low_confidence = sum(1 for h in hypotheses if h['confidence'] == 'Low')
        
        report += "CONFIDENCE DISTRIBUTION:\n"
        report += f"High Confidence: {high_confidence}\n"
        report += f"Medium Confidence: {medium_confidence}\n"
        report += f"Low Confidence: {low_confidence}\n\n"
        
        # Event type distribution
        event_types = {}
        for h in hypotheses:
            if h['closest_event']:
                event_type = h['event_type']
                event_types[event_type] = event_types.get(event_type, 0) + 1
        
        report += "EVENT TYPE DISTRIBUTION:\n"
        for event_type, count in event_types.items():
            report += f"{event_type}: {count}\n"
        report += "\n"
        
        # Detailed hypotheses
        report += "DETAILED HYPOTHESES:\n"
        report += "-" * 30 + "\n\n"
        
        for i, hypothesis in enumerate(hypotheses, 1):
            report += f"{i}. Change Point: {hypothesis['change_point_date'].strftime('%Y-%m-%d')}\n"
            report += f"   Confidence: {hypothesis['confidence']}\n"
            report += f"   Hypothesis: {hypothesis['hypothesis']}\n"
            
            if hypothesis['closest_event']:
                report += f"   Closest Event: {hypothesis['closest_event']['event']}\n"
                report += f"   Event Date: {hypothesis['closest_event']['date']}\n"
                report += f"   Days Difference: {hypothesis['days_difference']}\n"
                report += f"   Event Impact: {hypothesis['closest_event']['impact']}\n"
            
            report += "\n"
        
        # Limitations and caveats
        report += "LIMITATIONS AND CAVEATS:\n"
        report += "-" * 25 + "\n"
        report += "1. Statistical correlation does not imply causation\n"
        report += "2. Change points may be influenced by multiple factors\n"
        report += "3. Some events may have delayed effects on oil prices\n"
        report += "4. Market expectations and anticipatory effects not captured\n"
        report += "5. External factors beyond modeled events may influence prices\n\n"
        
        if output_file:
            with open(output_file, 'w') as f:
                f.write(report)
            print(f"Report saved to {output_file}")
        
        return report

def quantify_event_impact(change_point_date, before_data, after_data, event_name):
    """
    Quantify the impact of a specific event on oil prices.
    
    Parameters:
    -----------
    change_point_date : datetime
        Date of the change point
    before_data : array-like
        Price data before the change point
    after_data : array-like
        Price data after the change point
    event_name : str
        Name of the associated event
        
    Returns:
    --------
    str : Formatted impact description
    """
    before_mean = np.mean(before_data)
    after_mean = np.mean(after_data)
    
    change_amount = after_mean - before_mean
    percent_change = (change_amount / before_mean) * 100 if before_mean != 0 else 0
    
    direction = "increase" if change_amount > 0 else "decrease"
    
    impact_description = (
        f"Following the {event_name} around {change_point_date.strftime('%Y-%m-%d')}, "
        f"the model detects a change point, with the average daily price shifting "
        f"from ${before_mean:.2f} to ${after_mean:.2f}, "
        f"an {direction} of {abs(percent_change):.1f}%."
    )
    
    return impact_description

if __name__ == "__main__":
    # Example usage
    events_file = "../../data/processed/events.csv"
    
    # Initialize associator
    associator = EventAssociator(events_file)
    
    # Example change points (these would come from the Bayesian model)
    example_change_points = [
        pd.to_datetime('2008-09-15'),
        pd.to_datetime('2020-03-06'),
        pd.to_datetime('2022-02-24')
    ]
    
    # Find associations
    associations = associator.find_associated_events(example_change_points)
    
    # Generate hypotheses
    hypotheses = associator.generate_hypotheses(associations)
    
    # Create timeline plot
    fig = associator.create_timeline_plot(example_change_points)
    plt.show()
    
    # Generate report
    report = associator.create_summary_report(hypotheses)
    print(report)