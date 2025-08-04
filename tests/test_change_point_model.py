"""
Comprehensive test suite for change point detection models
Tests core functionality, edge cases, and model performance
"""

import pytest
import numpy as np
import pandas as pd
from unittest.mock import patch, MagicMock
import sys
import os

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from change_point_model import ChangePointModel
from change_point_evaluator import ChangePointEvaluator, ChangePointResult

class TestChangePointModel:
    """Test suite for ChangePointModel class"""
    
    @pytest.fixture
    def sample_data(self):
        """Create sample time series data for testing"""
        dates = pd.date_range('2020-01-01', periods=100, freq='D')
        # Create data with known change point at index 50
        prices1 = np.random.normal(50, 5, 50)
        prices2 = np.random.normal(70, 8, 50)
        prices = np.concatenate([prices1, prices2])
        
        return pd.DataFrame({
            'date': dates,
            'price': prices
        })
    
    def test_initialization(self, sample_data):
        """Test model initialization"""
        model = ChangePointModel(sample_data)
        assert model.data is not None
        assert len(model.data) == 100
        assert 'date' in model.data.columns
        assert 'price' in model.data.columns
    
    def test_pelt_detection(self, sample_data):
        """Test PELT change point detection"""
        model = ChangePointModel(sample_data, method='pelt')
        results = model.detect_change_points(penalty=10.0)
        
        assert 'change_points' in results
        assert 'method' in results
        assert results['method'] == 'PELT'
        assert isinstance(results['change_points'], list)
    
    def test_binary_segmentation(self, sample_data):
        """Test Binary Segmentation method"""
        model = ChangePointModel(sample_data, method='binseg')
        results = model.detect_change_points(penalty=10.0)
        
        assert results['method'] == 'Binary Segmentation'
        assert 'change_points' in results
    
    def test_sliding_window(self, sample_data):
        """Test Sliding Window method"""
        model = ChangePointModel(sample_data, method='window')
        results = model.detect_change_points()
        
        assert results['method'] == 'Sliding Window'
        assert 'window_size' in results
    
    def test_empty_data(self):
        """Test handling of empty data"""
        empty_data = pd.DataFrame({'date': [], 'price': []})
        model = ChangePointModel(empty_data)
        
        with pytest.raises(Exception):
            model.detect_change_points()
    
    def test_single_point_data(self):
        """Test handling of single data point"""
        single_data = pd.DataFrame({
            'date': [pd.Timestamp('2020-01-01')],
            'price': [50.0]
        })
        model = ChangePointModel(single_data)
        results = model.detect_change_points()
        
        assert len(results.get('change_points', [])) == 0
    
    def test_expected_outputs(self, sample_data):
        """Test expected outputs structure"""
        model = ChangePointModel(sample_data)
        expected = model.get_expected_outputs()
        
        assert 'expected_outputs' in expected
        assert 'limitations' in expected
        assert 'interpretation_guidelines' in expected

class TestChangePointEvaluator:
    """Test suite for ChangePointEvaluator class"""
    
    @pytest.fixture
    def evaluator_data(self):
        """Create data for evaluator testing"""
        dates = pd.date_range('2020-01-01', periods=200, freq='D')
        # Two regime data
        prices1 = np.random.normal(40, 3, 100)
        prices2 = np.random.normal(60, 5, 100)
        prices = np.concatenate([prices1, prices2])
        
        return pd.DataFrame({
            'date': dates,
            'price': prices
        })
    
    def test_quantitative_analysis(self, evaluator_data):
        """Test quantitative impact analysis"""
        evaluator = ChangePointEvaluator(evaluator_data)
        change_points = [100]  # Known change point
        
        results = evaluator.quantitative_impact_analysis(change_points, 'PELT')
        
        assert len(results) == 1
        assert isinstance(results[0], ChangePointResult)
        assert results[0].method == 'PELT'
        assert results[0].index == 100
        assert results[0].mean_before != results[0].mean_after
    
    def test_method_comparison(self, evaluator_data):
        """Test comparison between methods"""
        evaluator = ChangePointEvaluator(evaluator_data)
        methods_results = {
            'PELT': [100],
            'Binary Segmentation': [98, 102],
            'Sliding Window': [101]
        }
        
        comparison = evaluator.compare_methods(methods_results)
        
        assert 'method_performance' in comparison
        assert 'consensus_points' in comparison
        assert len(comparison['method_performance']) == 3
    
    def test_false_positive_simulation(self, evaluator_data):
        """Test false positive rate calculation"""
        evaluator = ChangePointEvaluator(evaluator_data)
        
        # Mock the ChangePointModel to avoid actual computation
        with patch('change_point_evaluator.ChangePointModel') as mock_model:
            mock_instance = MagicMock()
            mock_instance.detect_change_points.return_value = {
                'change_points': [250],  # Simulated detection
                'method': 'PELT'
            }
            mock_model.return_value = mock_instance
            
            results = evaluator.simulate_false_positive_test(n_simulations=10)
            
            assert 'false_positive_rates' in results
            assert 'detection_accuracy' in results
            assert 'simulation_summary' in results
    
    def test_assumptions_validation(self, evaluator_data):
        """Test model assumptions validation"""
        evaluator = ChangePointEvaluator(evaluator_data)
        assumptions = evaluator.explain_model_assumptions()
        
        required_assumptions = ['stationarity', 'independence', 'normality', 
                              'homoscedasticity', 'noise_level']
        
        for assumption in required_assumptions:
            assert assumption in assumptions
            assert 'assumption' in assumptions[assumption]
            assert 'validation' in assumptions[assumption]

class TestIntegration:
    """Integration tests for complete workflow"""
    
    def test_complete_workflow(self):
        """Test complete analysis workflow"""
        # Create synthetic data
        dates = pd.date_range('2020-01-01', periods=300, freq='D')
        prices = np.concatenate([
            np.random.normal(45, 4, 100),  # Regime 1
            np.random.normal(65, 6, 100),  # Regime 2
            np.random.normal(55, 5, 100)   # Regime 3
        ])
        
        data = pd.DataFrame({'date': dates, 'price': prices})
        
        # Run change point detection
        model = ChangePointModel(data)
        results = model.detect_change_points()
        
        # Evaluate results
        evaluator = ChangePointEvaluator(data)
        if results.get('change_points'):
            analysis = evaluator.quantitative_impact_analysis(
                results['change_points'], results['method']
            )
            
            assert len(analysis) > 0
            for result in analysis:
                assert isinstance(result, ChangePointResult)
                assert result.confidence > 0
                assert result.statistical_significance >= 0

if __name__ == '__main__':
    pytest.main([__file__, '-v'])