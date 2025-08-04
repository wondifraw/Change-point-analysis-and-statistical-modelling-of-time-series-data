"""
Data Preparation Module for Bayesian Change Point Analysis

This module handles data loading, preprocessing, and exploratory data analysis
for the Brent oil price change point detection.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

class DataPreprocessor:
    """
    Handles data preprocessing for change point analysis.
    """
    
    def __init__(self, data_path):
        """
        Initialize with data file path.
        
        Parameters:
        -----------
        data_path : str
            Path to the CSV file containing oil price data
        """
        self.data_path = data_path
        self.df = None
        self.prices = None
        self.log_returns = None
        self.dates = None
        
    def load_and_clean_data(self):
        """
        Load and clean the oil price data.
        
        Returns:
        --------
        pandas.DataFrame : Cleaned dataframe
        """
        # Read data
        self.df = pd.read_csv(self.data_path)
        
        # Handle different date formats in the dataset
        def parse_date(date_str):
            try:
                # Try format: 20-May-87
                return pd.to_datetime(date_str, format='%d-%b-%y')
            except:
                try:
                    # Try format: Apr 22, 2020
                    return pd.to_datetime(date_str, format='%b %d, %Y')
                except:
                    # Fallback to pandas auto-detection
                    return pd.to_datetime(date_str)
        
        self.df['Date'] = self.df['Date'].apply(parse_date)
        
        # Sort by date
        self.df = self.df.sort_values('Date').reset_index(drop=True)
        
        # Remove any duplicates
        self.df = self.df.drop_duplicates(subset=['Date']).reset_index(drop=True)
        
        # Extract arrays for analysis
        self.dates = self.df['Date'].values
        self.prices = self.df['Price'].values
        
        # Calculate log returns (more suitable for change point analysis)
        self.log_returns = np.diff(np.log(self.prices))
        
        print(f"Data loaded successfully:")
        print(f"- Date range: {self.df['Date'].min()} to {self.df['Date'].max()}")
        print(f"- Number of observations: {len(self.df)}")
        print(f"- Price range: ${self.prices.min():.2f} to ${self.prices.max():.2f}")
        
        return self.df
    
    def perform_eda(self, figsize=(15, 12)):
        """
        Perform exploratory data analysis.
        
        Parameters:
        -----------
        figsize : tuple
            Figure size for plots
            
        Returns:
        --------
        dict : EDA results and statistics
        """
        if self.df is None:
            raise ValueError("Data not loaded. Call load_and_clean_data() first.")
        
        fig, axes = plt.subplots(2, 3, figsize=figsize)
        
        # Plot 1: Price time series
        axes[0, 0].plot(self.df['Date'], self.df['Price'], linewidth=1)
        axes[0, 0].set_title('Brent Oil Prices Over Time')
        axes[0, 0].set_xlabel('Date')
        axes[0, 0].set_ylabel('Price (USD)')
        axes[0, 0].grid(True, alpha=0.3)
        
        # Plot 2: Log returns time series
        axes[0, 1].plot(self.df['Date'][1:], self.log_returns, linewidth=0.5, alpha=0.7)
        axes[0, 1].set_title('Log Returns Over Time')
        axes[0, 1].set_xlabel('Date')
        axes[0, 1].set_ylabel('Log Returns')
        axes[0, 1].grid(True, alpha=0.3)
        
        # Plot 3: Price distribution
        axes[0, 2].hist(self.prices, bins=50, density=True, alpha=0.7, color='skyblue')
        axes[0, 2].set_title('Price Distribution')
        axes[0, 2].set_xlabel('Price (USD)')
        axes[0, 2].set_ylabel('Density')
        axes[0, 2].grid(True, alpha=0.3)
        
        # Plot 4: Log returns distribution
        axes[1, 0].hist(self.log_returns, bins=50, density=True, alpha=0.7, color='lightgreen')
        axes[1, 0].set_title('Log Returns Distribution')
        axes[1, 0].set_xlabel('Log Returns')
        axes[1, 0].set_ylabel('Density')
        axes[1, 0].grid(True, alpha=0.3)
        
        # Plot 5: Rolling volatility
        window = 252  # Approximately 1 year of trading days
        rolling_std = pd.Series(self.log_returns).rolling(window=window).std()
        axes[1, 1].plot(self.df['Date'][window:], rolling_std[window:])
        axes[1, 1].set_title(f'Rolling Volatility ({window}-day window)')
        axes[1, 1].set_xlabel('Date')
        axes[1, 1].set_ylabel('Volatility')
        axes[1, 1].grid(True, alpha=0.3)
        
        # Plot 6: Q-Q plot for log returns
        stats.probplot(self.log_returns, dist="norm", plot=axes[1, 2])
        axes[1, 2].set_title('Q-Q Plot: Log Returns vs Normal')
        axes[1, 2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Calculate statistics
        eda_stats = {
            'price_stats': {
                'mean': np.mean(self.prices),
                'std': np.std(self.prices),
                'min': np.min(self.prices),
                'max': np.max(self.prices),
                'skewness': stats.skew(self.prices),
                'kurtosis': stats.kurtosis(self.prices)
            },
            'log_returns_stats': {
                'mean': np.mean(self.log_returns),
                'std': np.std(self.log_returns),
                'min': np.min(self.log_returns),
                'max': np.max(self.log_returns),
                'skewness': stats.skew(self.log_returns),
                'kurtosis': stats.kurtosis(self.log_returns)
            },
            'stationarity_test': self.test_stationarity(),
            'normality_test': self.test_normality()
        }
        
        return eda_stats, fig
    
    def test_stationarity(self):
        """
        Test for stationarity using Augmented Dickey-Fuller test.
        
        Returns:
        --------
        dict : Stationarity test results
        """
        from statsmodels.tsa.stattools import adfuller
        
        # Test prices
        adf_prices = adfuller(self.prices)
        
        # Test log returns
        adf_returns = adfuller(self.log_returns)
        
        results = {
            'prices': {
                'adf_statistic': adf_prices[0],
                'p_value': adf_prices[1],
                'is_stationary': adf_prices[1] < 0.05,
                'critical_values': adf_prices[4]
            },
            'log_returns': {
                'adf_statistic': adf_returns[0],
                'p_value': adf_returns[1],
                'is_stationary': adf_returns[1] < 0.05,
                'critical_values': adf_returns[4]
            }
        }
        
        return results
    
    def test_normality(self):
        """
        Test for normality using Shapiro-Wilk and Jarque-Bera tests.
        
        Returns:
        --------
        dict : Normality test results
        """
        # Shapiro-Wilk test (use sample if data is too large)
        sample_size = min(5000, len(self.log_returns))
        sample_returns = np.random.choice(self.log_returns, sample_size, replace=False)
        
        shapiro_stat, shapiro_p = stats.shapiro(sample_returns)
        
        # Jarque-Bera test
        jb_stat, jb_p = stats.jarque_bera(self.log_returns)
        
        results = {
            'shapiro_wilk': {
                'statistic': shapiro_stat,
                'p_value': shapiro_p,
                'is_normal': shapiro_p > 0.05,
                'sample_size': sample_size
            },
            'jarque_bera': {
                'statistic': jb_stat,
                'p_value': jb_p,
                'is_normal': jb_p > 0.05
            }
        }
        
        return results
    
    def detect_outliers(self, method='iqr', threshold=3):
        """
        Detect outliers in the data.
        
        Parameters:
        -----------
        method : str
            Method for outlier detection ('iqr' or 'zscore')
        threshold : float
            Threshold for outlier detection
            
        Returns:
        --------
        dict : Outlier detection results
        """
        if method == 'iqr':
            Q1 = np.percentile(self.log_returns, 25)
            Q3 = np.percentile(self.log_returns, 75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            outliers = (self.log_returns < lower_bound) | (self.log_returns > upper_bound)
            
        elif method == 'zscore':
            z_scores = np.abs(stats.zscore(self.log_returns))
            outliers = z_scores > threshold
        
        outlier_indices = np.where(outliers)[0]
        outlier_dates = self.dates[1:][outlier_indices]  # +1 because log_returns is shorter
        outlier_values = self.log_returns[outlier_indices]
        
        results = {
            'method': method,
            'threshold': threshold,
            'num_outliers': len(outlier_indices),
            'outlier_percentage': (len(outlier_indices) / len(self.log_returns)) * 100,
            'outlier_indices': outlier_indices,
            'outlier_dates': outlier_dates,
            'outlier_values': outlier_values
        }
        
        return results
    
    def create_summary_report(self, eda_stats, outlier_results):
        """
        Create a comprehensive summary report.
        
        Parameters:
        -----------
        eda_stats : dict
            Results from perform_eda()
        outlier_results : dict
            Results from detect_outliers()
            
        Returns:
        --------
        str : Summary report
        """
        report = "BRENT OIL PRICE DATA ANALYSIS REPORT\n"
        report += "=" * 40 + "\n\n"
        
        # Basic statistics
        report += "BASIC STATISTICS:\n"
        report += f"Data period: {self.df['Date'].min().strftime('%Y-%m-%d')} to {self.df['Date'].max().strftime('%Y-%m-%d')}\n"
        report += f"Number of observations: {len(self.df)}\n"
        report += f"Price range: ${eda_stats['price_stats']['min']:.2f} - ${eda_stats['price_stats']['max']:.2f}\n"
        report += f"Average price: ${eda_stats['price_stats']['mean']:.2f}\n"
        report += f"Price volatility (std): ${eda_stats['price_stats']['std']:.2f}\n\n"
        
        # Log returns statistics
        report += "LOG RETURNS STATISTICS:\n"
        report += f"Mean: {eda_stats['log_returns_stats']['mean']:.6f}\n"
        report += f"Standard deviation: {eda_stats['log_returns_stats']['std']:.6f}\n"
        report += f"Skewness: {eda_stats['log_returns_stats']['skewness']:.4f}\n"
        report += f"Kurtosis: {eda_stats['log_returns_stats']['kurtosis']:.4f}\n\n"
        
        # Stationarity tests
        report += "STATIONARITY TESTS (ADF):\n"
        price_stat = eda_stats['stationarity_test']['prices']
        returns_stat = eda_stats['stationarity_test']['log_returns']
        
        report += f"Prices: {'Stationary' if price_stat['is_stationary'] else 'Non-stationary'} (p-value: {price_stat['p_value']:.6f})\n"
        report += f"Log Returns: {'Stationary' if returns_stat['is_stationary'] else 'Non-stationary'} (p-value: {returns_stat['p_value']:.6f})\n\n"
        
        # Normality tests
        report += "NORMALITY TESTS:\n"
        shapiro = eda_stats['normality_test']['shapiro_wilk']
        jb = eda_stats['normality_test']['jarque_bera']
        
        report += f"Shapiro-Wilk: {'Normal' if shapiro['is_normal'] else 'Non-normal'} (p-value: {shapiro['p_value']:.6f})\n"
        report += f"Jarque-Bera: {'Normal' if jb['is_normal'] else 'Non-normal'} (p-value: {jb['p_value']:.6f})\n\n"
        
        # Outliers
        report += "OUTLIER ANALYSIS:\n"
        report += f"Method: {outlier_results['method'].upper()}\n"
        report += f"Number of outliers: {outlier_results['num_outliers']}\n"
        report += f"Percentage of outliers: {outlier_results['outlier_percentage']:.2f}%\n\n"
        
        # Modeling implications
        report += "MODELING IMPLICATIONS:\n"
        report += "- Log returns are more suitable for change point analysis (stationary)\n"
        report += "- Non-normal distribution suggests fat tails and volatility clustering\n"
        report += "- Outliers may indicate structural breaks or extreme events\n"
        report += "- Bayesian approach can handle uncertainty in change point location\n\n"
        
        return report
    
    def get_processed_data(self):
        """
        Get the processed data for change point analysis.
        
        Returns:
        --------
        tuple : (prices, log_returns, dates)
        """
        if self.df is None:
            raise ValueError("Data not loaded. Call load_and_clean_data() first.")
        
        return self.prices, self.log_returns, self.dates

if __name__ == "__main__":
    # Example usage
    data_path = "../../data/raw/brent_oil_prices.csv"
    
    # Initialize preprocessor
    preprocessor = DataPreprocessor(data_path)
    
    # Load and clean data
    df = preprocessor.load_and_clean_data()
    
    # Perform EDA
    eda_stats, eda_fig = preprocessor.perform_eda()
    plt.show()
    
    # Detect outliers
    outlier_results = preprocessor.detect_outliers(method='iqr')
    
    # Generate summary report
    report = preprocessor.create_summary_report(eda_stats, outlier_results)
    print(report)
    
    # Get processed data
    prices, log_returns, dates = preprocessor.get_processed_data()
    print(f"Processed data shapes: prices={prices.shape}, log_returns={log_returns.shape}, dates={dates.shape}")