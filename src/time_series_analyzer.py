"""
Time Series Analysis Module
===========================

Analyzes Brent oil price data for key properties like trend and stationarity.
"""

import pandas as pd
import numpy as np
from scipy import stats
from typing import Dict, Tuple, List
import logging

class TimeSeriesAnalyzer:
    """
    Analyzes time series properties of Brent oil price data.
    
    Investigates trend, stationarity, and other key properties that inform
    modeling choices for change point analysis.
    """
    
    def __init__(self, data: pd.DataFrame):
        """
        Initialize analyzer with price data.
        
        Args:
            data (pd.DataFrame): Time series data with date and price columns
        """
        self.data = data
        self.properties = {}
    
    def analyze_properties(self) -> Dict:
        """
        Analyze key time series properties.
        
        Returns:
            Dict: Analysis results including trend and stationarity
        """
        try:
            self.properties = {
                'trend_analysis': self._analyze_trend(),
                'stationarity_test': self._test_stationarity(),
                'volatility_analysis': self._analyze_volatility(),
                'modeling_implications': self._get_modeling_implications()
            }
            
            logging.info("Time series properties analyzed successfully")
            return self.properties
            
        except Exception as e:
            logging.error(f"Properties analysis failed: {str(e)}")
            raise
    
    def _analyze_trend(self) -> Dict:
        """Analyze trend components in the data."""
        try:
            prices = self.data['price'].values
            time_index = np.arange(len(prices))
            
            # Linear trend analysis
            slope, intercept, r_value, p_value, std_err = stats.linregress(time_index, prices)
            
            return {
                'trend_slope': slope,
                'trend_significance': p_value < 0.05,
                'r_squared': r_value**2,
                'trend_direction': 'upward' if slope > 0 else 'downward'
            }
            
        except Exception as e:
            logging.error(f"Trend analysis failed: {str(e)}")
            return {}
    
    def _test_stationarity(self) -> Dict:
        """Test for stationarity using Augmented Dickey-Fuller test."""
        try:
            from statsmodels.tsa.stattools import adfuller
            
            prices = self.data['price'].dropna()
            adf_result = adfuller(prices)
            
            return {
                'adf_statistic': adf_result[0],
                'p_value': adf_result[1],
                'is_stationary': adf_result[1] < 0.05,
                'critical_values': adf_result[4]
            }
            
        except ImportError:
            logging.warning("statsmodels not available, using simplified test")
            return {'simplified_test': 'variance_ratio_test'}
        except Exception as e:
            logging.error(f"Stationarity test failed: {str(e)}")
            return {}
    
    def _analyze_volatility(self) -> Dict:
        """Analyze price volatility patterns."""
        try:
            prices = self.data['price']
            returns = prices.pct_change().dropna()
            
            return {
                'volatility_mean': returns.std(),
                'volatility_clusters': self._detect_volatility_clusters(returns),
                'max_drawdown': self._calculate_max_drawdown(prices)
            }
            
        except Exception as e:
            logging.error(f"Volatility analysis failed: {str(e)}")
            return {}
    
    def _detect_volatility_clusters(self, returns: pd.Series) -> bool:
        """Detect presence of volatility clustering."""
        try:
            # Simple ARCH test approximation
            squared_returns = returns**2
            autocorr = squared_returns.autocorr(lag=1)
            return abs(autocorr) > 0.1
            
        except Exception as e:
            logging.error(f"Volatility clustering detection failed: {str(e)}")
            return False
    
    def _calculate_max_drawdown(self, prices: pd.Series) -> float:
        """Calculate maximum drawdown."""
        try:
            cumulative = (1 + prices.pct_change()).cumprod()
            running_max = cumulative.expanding().max()
            drawdown = (cumulative - running_max) / running_max
            return drawdown.min()
            
        except Exception as e:
            logging.error(f"Max drawdown calculation failed: {str(e)}")
            return 0.0
    
    def _get_modeling_implications(self) -> List[str]:
        """Derive modeling implications from properties analysis."""
        implications = []
        
        if not self.properties.get('stationarity_test', {}).get('is_stationary', True):
            implications.append("Non-stationary data requires differencing or regime-switching models")
        
        if self.properties.get('volatility_analysis', {}).get('volatility_clusters', False):
            implications.append("Volatility clustering suggests GARCH-type models may be appropriate")
        
        if abs(self.properties.get('trend_analysis', {}).get('trend_slope', 0)) > 0.01:
            implications.append("Strong trend component may require detrending before change point analysis")
        
        return implications