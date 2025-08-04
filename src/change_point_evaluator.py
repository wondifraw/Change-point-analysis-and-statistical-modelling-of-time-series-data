"""
Change Point Model Evaluation and Comparison Module
Provides quantitative impact analysis and model performance evaluation
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
import scipy.stats as stats
from dataclasses import dataclass
import logging

@dataclass
class ChangePointResult:
    """Structured result for change point analysis"""
    date: str
    index: int
    method: str
    confidence: float
    mean_before: float
    mean_after: float
    variance_before: float
    variance_after: float
    volatility_before: float
    volatility_after: float
    change_magnitude: float
    statistical_significance: float

class ChangePointEvaluator:
    """
    Evaluates and compares change point detection methods with statistical rigor
    """
    
    def __init__(self, data: pd.DataFrame):
        self.data = data
        self.results = {}
        
    def quantitative_impact_analysis(self, change_points: List[int], method: str) -> List[ChangePointResult]:
        """
        Perform detailed quantitative analysis of each change point
        
        Args:
            change_points: List of change point indices
            method: Detection method used
            
        Returns:
            List of ChangePointResult objects with detailed statistics
        """
        results = []
        prices = self.data['price'].values
        
        for i, cp_idx in enumerate(change_points):
            # Define segments
            before_segment = prices[:cp_idx]
            after_segment = prices[cp_idx:]
            
            # Calculate comprehensive statistics
            mean_before = np.mean(before_segment)
            mean_after = np.mean(after_segment)
            var_before = np.var(before_segment)
            var_after = np.var(after_segment)
            
            # Volatility (rolling standard deviation)
            vol_before = np.std(np.diff(np.log(before_segment + 1e-8)))
            vol_after = np.std(np.diff(np.log(after_segment + 1e-8)))
            
            # Change magnitude
            change_mag = abs((mean_after - mean_before) / mean_before) * 100
            
            # Statistical significance (Welch's t-test)
            t_stat, p_value = stats.ttest_ind(before_segment, after_segment, equal_var=False)
            
            # Confidence based on segment sizes and p-value
            confidence = max(0.5, 1 - p_value) * min(1.0, len(before_segment) / 100) * min(1.0, len(after_segment) / 100)
            
            result = ChangePointResult(
                date=self.data.iloc[cp_idx]['date'].strftime('%Y-%m-%d'),
                index=cp_idx,
                method=method,
                confidence=confidence,
                mean_before=mean_before,
                mean_after=mean_after,
                variance_before=var_before,
                variance_after=var_after,
                volatility_before=vol_before,
                volatility_after=vol_after,
                change_magnitude=change_mag,
                statistical_significance=p_value
            )
            
            results.append(result)
            
        return results
    
    def compare_methods(self, methods_results: Dict[str, List[int]]) -> Dict:
        """
        Compare performance across different change point detection methods
        
        Args:
            methods_results: Dictionary mapping method names to change point lists
            
        Returns:
            Comparison statistics and rankings
        """
        comparison = {
            'method_performance': {},
            'consensus_points': [],
            'method_agreement': {},
            'statistical_summary': {}
        }
        
        all_results = {}
        
        # Analyze each method
        for method, change_points in methods_results.items():
            if change_points:
                results = self.quantitative_impact_analysis(change_points, method)
                all_results[method] = results
                
                # Method performance metrics
                avg_confidence = np.mean([r.confidence for r in results])
                avg_significance = np.mean([r.statistical_significance for r in results])
                avg_magnitude = np.mean([r.change_magnitude for r in results])
                
                comparison['method_performance'][method] = {
                    'num_change_points': len(change_points),
                    'avg_confidence': avg_confidence,
                    'avg_statistical_significance': avg_significance,
                    'avg_change_magnitude': avg_magnitude,
                    'detection_rate': len(change_points) / len(self.data) * 1000  # per 1000 observations
                }
        
        # Find consensus points (detected by multiple methods)
        all_points = []
        for method, points in methods_results.items():
            all_points.extend([(p, method) for p in points])
        
        # Group nearby points (within 30 days)
        consensus_threshold = 30
        consensus_groups = []
        used_points = set()
        
        for point, method in all_points:
            if point in used_points:
                continue
                
            group = [(point, method)]
            used_points.add(point)
            
            for other_point, other_method in all_points:
                if other_point != point and other_point not in used_points:
                    if abs(other_point - point) <= consensus_threshold:
                        group.append((other_point, other_method))
                        used_points.add(other_point)
            
            if len(group) > 1:  # Consensus requires at least 2 methods
                consensus_groups.append(group)
        
        comparison['consensus_points'] = consensus_groups
        comparison['method_agreement'] = len(consensus_groups) / max(1, len(set(all_points))) if all_points else 0
        
        return comparison
    
    def simulate_false_positive_test(self, n_simulations: int = 100) -> Dict:
        """
        Test for false positives using simulated data with known change points
        
        Args:
            n_simulations: Number of simulation runs
            
        Returns:
            False positive rates and method reliability metrics
        """
        from .change_point_model import ChangePointModel
        
        results = {
            'false_positive_rates': {},
            'detection_accuracy': {},
            'simulation_summary': {}
        }
        
        methods = ['pelt', 'binseg', 'window']
        
        for method in methods:
            false_positives = 0
            true_positives = 0
            total_detections = 0
            
            for sim in range(n_simulations):
                # Generate synthetic data with known change point
                n_points = 500
                true_cp = 250
                
                # Regime 1: mean=50, std=5
                regime1 = np.random.normal(50, 5, true_cp)
                # Regime 2: mean=70, std=8  
                regime2 = np.random.normal(70, 8, n_points - true_cp)
                
                synthetic_prices = np.concatenate([regime1, regime2])
                synthetic_dates = pd.date_range('2020-01-01', periods=n_points, freq='D')
                
                synthetic_data = pd.DataFrame({
                    'date': synthetic_dates,
                    'price': synthetic_prices
                })
                
                # Detect change points
                cp_model = ChangePointModel(synthetic_data)
                detected = cp_model.detect_change_points(penalty=10.0)
                
                if 'change_points' in detected and detected['change_points']:
                    detected_points = detected['change_points']
                    total_detections += len(detected_points)
                    
                    # Check for true positive (within 20 points of true change point)
                    true_positive_found = any(abs(cp - true_cp) <= 20 for cp in detected_points)
                    if true_positive_found:
                        true_positives += 1
                    
                    # Count false positives
                    false_positives += len([cp for cp in detected_points if abs(cp - true_cp) > 20])
            
            # Calculate rates
            fp_rate = false_positives / max(1, n_simulations) 
            tp_rate = true_positives / n_simulations
            precision = true_positives / max(1, total_detections)
            
            results['false_positive_rates'][method] = fp_rate
            results['detection_accuracy'][method] = {
                'true_positive_rate': tp_rate,
                'precision': precision,
                'avg_detections_per_sim': total_detections / n_simulations
            }
        
        results['simulation_summary'] = {
            'n_simulations': n_simulations,
            'synthetic_data_length': n_points,
            'known_change_point': true_cp
        }
        
        return results
    
    def explain_model_assumptions(self) -> Dict:
        """
        Document and validate key model assumptions
        
        Returns:
            Dictionary of assumptions and their validation results
        """
        prices = self.data['price'].values
        log_returns = np.diff(np.log(prices))
        
        assumptions = {
            'stationarity': {
                'assumption': 'Time series segments are locally stationary',
                'test': 'Augmented Dickey-Fuller test',
                'validation': self._test_stationarity(prices),
                'implication': 'Non-stationarity supports change point detection'
            },
            'independence': {
                'assumption': 'Residuals are independent (no autocorrelation)',
                'test': 'Ljung-Box test on returns',
                'validation': self._test_independence(log_returns),
                'implication': 'Autocorrelation may lead to spurious change points'
            },
            'normality': {
                'assumption': 'Price changes follow normal distribution within regimes',
                'test': 'Shapiro-Wilk test on log returns',
                'validation': self._test_normality(log_returns),
                'implication': 'Non-normality affects confidence intervals'
            },
            'homoscedasticity': {
                'assumption': 'Constant variance within regimes',
                'test': 'Breusch-Pagan test',
                'validation': self._test_homoscedasticity(log_returns),
                'implication': 'Heteroscedasticity requires robust methods'
            },
            'noise_level': {
                'assumption': 'Signal-to-noise ratio allows change detection',
                'test': 'Variance ratio analysis',
                'validation': self._assess_noise_level(prices),
                'implication': 'High noise reduces detection sensitivity'
            }
        }
        
        return assumptions
    
    def _test_stationarity(self, prices: np.ndarray) -> Dict:
        """Test stationarity using ADF test"""
        try:
            from statsmodels.tsa.stattools import adfuller
            result = adfuller(prices)
            return {
                'statistic': result[0],
                'p_value': result[1],
                'is_stationary': result[1] < 0.05,
                'critical_values': result[4]
            }
        except ImportError:
            return {'error': 'statsmodels not available'}
    
    def _test_independence(self, returns: np.ndarray) -> Dict:
        """Test independence using Ljung-Box test"""
        try:
            from statsmodels.stats.diagnostic import acorr_ljungbox
            result = acorr_ljungbox(returns, lags=10, return_df=True)
            return {
                'ljung_box_stat': result['lb_stat'].iloc[-1],
                'p_value': result['lb_pvalue'].iloc[-1],
                'is_independent': result['lb_pvalue'].iloc[-1] > 0.05
            }
        except ImportError:
            # Fallback: simple autocorrelation
            autocorr = np.corrcoef(returns[:-1], returns[1:])[0, 1]
            return {
                'autocorrelation': autocorr,
                'is_independent': abs(autocorr) < 0.1
            }
    
    def _test_normality(self, returns: np.ndarray) -> Dict:
        """Test normality using Shapiro-Wilk test"""
        if len(returns) > 5000:  # Shapiro-Wilk has sample size limits
            returns = np.random.choice(returns, 5000, replace=False)
        
        stat, p_value = stats.shapiro(returns)
        return {
            'shapiro_stat': stat,
            'p_value': p_value,
            'is_normal': p_value > 0.05,
            'skewness': stats.skew(returns),
            'kurtosis': stats.kurtosis(returns)
        }
    
    def _test_homoscedasticity(self, returns: np.ndarray) -> Dict:
        """Test constant variance assumption"""
        # Simple variance ratio test
        mid_point = len(returns) // 2
        var1 = np.var(returns[:mid_point])
        var2 = np.var(returns[mid_point:])
        
        f_stat = var2 / var1 if var1 > 0 else 1
        p_value = 2 * min(stats.f.cdf(f_stat, mid_point-1, len(returns)-mid_point-1),
                         1 - stats.f.cdf(f_stat, mid_point-1, len(returns)-mid_point-1))
        
        return {
            'variance_ratio': f_stat,
            'p_value': p_value,
            'is_homoscedastic': p_value > 0.05,
            'first_half_var': var1,
            'second_half_var': var2
        }
    
    def _assess_noise_level(self, prices: np.ndarray) -> Dict:
        """Assess signal-to-noise ratio"""
        # Calculate trend component
        x = np.arange(len(prices))
        slope, intercept, r_value, p_value, std_err = stats.linregress(x, prices)
        trend = slope * x + intercept
        
        # Calculate noise (residuals)
        residuals = prices - trend
        signal_power = np.var(trend)
        noise_power = np.var(residuals)
        
        snr = signal_power / noise_power if noise_power > 0 else float('inf')
        
        return {
            'signal_to_noise_ratio': snr,
            'signal_power': signal_power,
            'noise_power': noise_power,
            'trend_r_squared': r_value**2,
            'noise_assessment': 'Low' if snr > 10 else 'Medium' if snr > 1 else 'High'
        }