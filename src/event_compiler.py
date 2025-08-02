"""
Event Data Compiler Module
==========================

Handles research and compilation of geopolitical events, OPEC decisions,
and economic shocks relevant to oil market analysis.
"""

import pandas as pd
from datetime import datetime
from typing import List, Dict, Optional
import logging

class EventCompiler:
    """
    Compiles and manages geopolitical and economic events affecting oil prices.
    
    This class handles the research, compilation, and structuring of major events
    that could impact Brent oil prices for change point analysis.
    """
    
    def __init__(self, output_path: str = "data/processed/events.csv"):
        """
        Initialize the event compiler.
        
        Args:
            output_path (str): Path to save compiled events dataset
        """
        self.output_path = output_path
        self.events_data = []
        
    def compile_major_events(self) -> pd.DataFrame:
        """
        Compile major geopolitical and economic events affecting oil markets.
        
        Returns:
            pd.DataFrame: Structured dataset of events with dates and descriptions
        """
        try:
            # Create output directory if it doesn't exist
            import os
            os.makedirs(os.path.dirname(self.output_path), exist_ok=True)
            # Define major oil market events (15+ key events)
            events = [
                {"date": "1990-08-02", "event": "Iraq invasion of Kuwait", "category": "Geopolitical", "impact": "High"},
                {"date": "2001-09-11", "event": "September 11 attacks", "category": "Geopolitical", "impact": "High"},
                {"date": "2003-03-20", "event": "Iraq War begins", "category": "Geopolitical", "impact": "High"},
                {"date": "2008-09-15", "event": "Lehman Brothers collapse", "category": "Economic", "impact": "High"},
                {"date": "2010-12-17", "event": "Arab Spring begins", "category": "Geopolitical", "impact": "Medium"},
                {"date": "2011-03-19", "event": "Libya civil war intervention", "category": "Geopolitical", "impact": "Medium"},
                {"date": "2014-11-27", "event": "OPEC maintains production", "category": "OPEC Decision", "impact": "High"},
                {"date": "2016-11-30", "event": "OPEC production cut agreement", "category": "OPEC Decision", "impact": "High"},
                {"date": "2018-05-08", "event": "US withdraws from Iran nuclear deal", "category": "Geopolitical", "impact": "Medium"},
                {"date": "2020-03-06", "event": "OPEC+ deal collapse", "category": "OPEC Decision", "impact": "High"},
                {"date": "2020-03-11", "event": "WHO declares COVID-19 pandemic", "category": "Economic", "impact": "High"},
                {"date": "2020-04-20", "event": "WTI oil futures turn negative", "category": "Economic", "impact": "High"},
                {"date": "2022-02-24", "event": "Russia invades Ukraine", "category": "Geopolitical", "impact": "High"},
                {"date": "2022-10-05", "event": "OPEC+ announces production cuts", "category": "OPEC Decision", "impact": "Medium"},
                {"date": "2023-04-02", "event": "OPEC+ surprise production cuts", "category": "OPEC Decision", "impact": "Medium"}
            ]
            
            df = pd.DataFrame(events)
            df['date'] = pd.to_datetime(df['date'])
            df = df.sort_values('date').reset_index(drop=True)
            
            # Save to CSV
            df.to_csv(self.output_path, index=False)
            logging.info(f"Compiled {len(df)} events and saved to {self.output_path}")
            
            return df
            
        except Exception as e:
            logging.error(f"Event compilation failed: {str(e)}")
            raise
    
    def add_custom_event(self, date: str, event: str, category: str, impact: str) -> None:
        """
        Add a custom event to the dataset.
        
        Args:
            date (str): Event date in YYYY-MM-DD format
            event (str): Event description
            category (str): Event category (Geopolitical, Economic, OPEC Decision)
            impact (str): Expected impact level (High, Medium, Low)
        """
        try:
            new_event = {
                "date": pd.to_datetime(date),
                "event": event,
                "category": category,
                "impact": impact
            }
            self.events_data.append(new_event)
            logging.info(f"Added custom event: {event}")
            
        except Exception as e:
            logging.error(f"Failed to add custom event: {str(e)}")
            raise
    
    def validate_events_data(self, df: pd.DataFrame) -> Dict[str, bool]:
        """
        Validate the compiled events dataset.
        
        Args:
            df (pd.DataFrame): Events dataset to validate
            
        Returns:
            Dict[str, bool]: Validation results
        """
        try:
            validation = {
                'has_minimum_events': len(df) >= 10,
                'dates_valid': df['date'].notna().all(),
                'required_columns': all(col in df.columns for col in ['date', 'event', 'category', 'impact']),
                'chronological_order': df['date'].is_monotonic_increasing
            }
            
            logging.info(f"Events validation completed: {validation}")
            return validation
            
        except Exception as e:
            logging.error(f"Events validation failed: {str(e)}")
            return {'validation_error': True}