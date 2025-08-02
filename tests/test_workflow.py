"""
Basic tests for the analysis workflow
"""
import pytest
import pandas as pd
import numpy as np
from src.data_workflow import DataAnalysisWorkflow
from src.event_compiler import EventCompiler
from src.time_series_analyzer import TimeSeriesAnalyzer
from src.change_point_model import ChangePointModel

def test_event_compiler():
    """Test event compilation"""
    compiler = EventCompiler()
    events_df = compiler.compile_major_events()
    assert len(events_df) >= 10
    assert 'date' in events_df.columns
    assert 'event' in events_df.columns

def test_time_series_analyzer():
    """Test time series analysis"""
    # Create synthetic data
    dates = pd.date_range('2000-01-01', '2023-12-31', freq='D')
    prices = 50 + np.random.randn(len(dates)).cumsum() * 2
    data = pd.DataFrame({'date': dates, 'price': prices})
    
    analyzer = TimeSeriesAnalyzer(data)
    properties = analyzer.analyze_properties()
    
    assert 'trend_analysis' in properties
    assert 'stationarity_test' in properties

def test_change_point_model():
    """Test change point detection"""
    # Create synthetic data
    dates = pd.date_range('2000-01-01', '2023-12-31', freq='D')
    prices = 50 + np.random.randn(len(dates)).cumsum() * 2
    data = pd.DataFrame({'date': dates, 'price': prices})
    
    model = ChangePointModel(data, method='pelt')
    results = model.detect_change_points()
    
    assert 'change_points' in results
    assert 'method' in results