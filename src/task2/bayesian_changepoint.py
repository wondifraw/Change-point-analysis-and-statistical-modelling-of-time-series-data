"""
Bayesian Change Point Detection Model for Brent Oil Prices

This module implements a Bayesian change point detection model using PyMC3
to identify statistically significant structural breaks in the Brent oil price series.
"""

import numpy as np
import pandas as pd
import pymc3 as pm
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

class BayesianChangePointModel:
    """
    Bayesian Change Point Detection Model for time series analysis.
    
    This model identifies structural breaks in time series data by modeling
    a switch point where the statistical properties (mean, variance) change.
    """
    
    def __init__(self, data, dates):
        """
        Initialize the model with data and dates.
        
        Parameters:
        -----------
        data : array-like
            Time series data (e.g., oil prices or log returns)
        dates : array-like
            Corresponding dates for the time series
        """
        self.data = np.array(data)
        self.dates = pd.to_datetime(dates)
        self.n_obs = len(data)
        self.model = None
        self.trace = None
        
    def build_model(self, model_type='mean_change'):
        """
        Build the Bayesian change point model.
        
        Parameters:
        -----------
        model_type : str
            Type of change point model ('mean_change' or 'variance_change')
        """
        with pm.Model() as model:
            # Switch point - uniform prior over all possible time points
            tau = pm.DiscreteUniform('tau', lower=1, upper=self.n_obs-2)
            
            if model_type == 'mean_change':
                # Parameters for before and after the change point
                mu_1 = pm.Normal('mu_1', mu=0, sigma=10)  # Mean before change
                mu_2 = pm.Normal('mu_2', mu=0, sigma=10)  # Mean after change
                sigma = pm.HalfNormal('sigma', sigma=5)   # Common variance
                
                # Switch function to select appropriate mean
                mu = pm.math.switch(tau >= np.arange(self.n_obs), mu_1, mu_2)
                
                # Likelihood
                obs = pm.Normal('obs', mu=mu, sigma=sigma, observed=self.data)
                
            elif model_type == 'variance_change':
                # Parameters for variance change model
                mu = pm.Normal('mu', mu=0, sigma=10)      # Common mean
                sigma_1 = pm.HalfNormal('sigma_1', sigma=5)  # Variance before change
                sigma_2 = pm.HalfNormal('sigma_2', sigma=5)  # Variance after change
                
                # Switch function to select appropriate variance
                sigma = pm.math.switch(tau >= np.arange(self.n_obs), sigma_1, sigma_2)
                
                # Likelihood
                obs = pm.Normal('obs', mu=mu, sigma=sigma, observed=self.data)
        
        self.model = model
        return model
    
    def fit_model(self, draws=2000, tune=1000, chains=2):
        """
        Fit the Bayesian model using MCMC sampling.
        
        Parameters:
        -----------
        draws : int
            Number of samples to draw
        tune : int
            Number of tuning samples
        chains : int
            Number of MCMC chains
        """
        if self.model is None:
            raise ValueError("Model not built. Call build_model() first.")
        
        with self.model:
            self.trace = pm.sample(draws=draws, tune=tune, chains=chains, 
                                 return_inferencedata=False, progressbar=True)
        
        return self.trace
    
    def get_change_point_summary(self):
        """
        Get summary statistics for the change point.
        
        Returns:
        --------
        dict : Summary statistics including most probable change point date
        """
        if self.trace is None:
            raise ValueError("Model not fitted. Call fit_model() first.")
        
        tau_samples = self.trace['tau']
        
        # Most probable change point
        tau_mode = int(np.round(np.mean(tau_samples)))
        change_date = self.dates[tau_mode]
        
        # Credible interval
        tau_hdi = pm.hdi(tau_samples, hdi_prob=0.95)
        change_date_lower = self.dates[int(tau_hdi[0])]
        change_date_upper = self.dates[int(tau_hdi[1])]
        
        summary = {
            'most_probable_change_point': tau_mode,
            'change_date': change_date,
            'credible_interval_lower': change_date_lower,
            'credible_interval_upper': change_date_upper,
            'tau_mean': np.mean(tau_samples),
            'tau_std': np.std(tau_samples)
        }
        
        return summary
    
    def get_parameter_summary(self):
        """
        Get summary statistics for model parameters.
        
        Returns:
        --------
        dict : Summary statistics for all model parameters
        """
        if self.trace is None:
            raise ValueError("Model not fitted. Call fit_model() first.")
        
        summary = {}
        
        for var_name in self.trace.varnames:
            if var_name != 'obs':
                samples = self.trace[var_name]
                summary[var_name] = {
                    'mean': np.mean(samples),
                    'std': np.std(samples),
                    'hdi_95': pm.hdi(samples, hdi_prob=0.95)
                }
        
        return summary
    
    def plot_results(self, figsize=(15, 10)):
        """
        Plot the results of the change point analysis.
        
        Parameters:
        -----------
        figsize : tuple
            Figure size for the plots
        """
        if self.trace is None:
            raise ValueError("Model not fitted. Call fit_model() first.")
        
        fig, axes = plt.subplots(2, 2, figsize=figsize)
        
        # Plot 1: Time series with change point
        ax1 = axes[0, 0]
        ax1.plot(self.dates, self.data, 'b-', alpha=0.7, label='Data')
        
        # Add change point
        summary = self.get_change_point_summary()
        change_date = summary['change_date']
        ax1.axvline(change_date, color='red', linestyle='--', linewidth=2, 
                   label=f'Change Point: {change_date.strftime("%Y-%m-%d")}')
        
        # Add credible interval
        ax1.axvspan(summary['credible_interval_lower'], 
                   summary['credible_interval_upper'], 
                   alpha=0.2, color='red', label='95% Credible Interval')
        
        ax1.set_title('Time Series with Detected Change Point')
        ax1.set_xlabel('Date')
        ax1.set_ylabel('Value')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Posterior distribution of tau
        ax2 = axes[0, 1]
        tau_samples = self.trace['tau']
        ax2.hist(tau_samples, bins=50, density=True, alpha=0.7, color='skyblue')
        ax2.axvline(summary['most_probable_change_point'], color='red', 
                   linestyle='--', label='Most Probable')
        ax2.set_title('Posterior Distribution of Change Point')
        ax2.set_xlabel('Time Index')
        ax2.set_ylabel('Density')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Plot 3: Parameter traces
        ax3 = axes[1, 0]
        if 'mu_1' in self.trace.varnames:
            ax3.plot(self.trace['mu_1'], alpha=0.7, label='μ₁ (before)')
            ax3.plot(self.trace['mu_2'], alpha=0.7, label='μ₂ (after)')
        ax3.set_title('Parameter Traces')
        ax3.set_xlabel('Iteration')
        ax3.set_ylabel('Parameter Value')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # Plot 4: Before/After comparison
        ax4 = axes[1, 1]
        change_idx = summary['most_probable_change_point']
        
        before_data = self.data[:change_idx]
        after_data = self.data[change_idx:]
        
        ax4.hist(before_data, bins=30, alpha=0.7, label='Before Change', density=True)
        ax4.hist(after_data, bins=30, alpha=0.7, label='After Change', density=True)
        ax4.set_title('Distribution Before vs After Change Point')
        ax4.set_xlabel('Value')
        ax4.set_ylabel('Density')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig
    
    def quantify_impact(self):
        """
        Quantify the impact of the change point on the time series.
        
        Returns:
        --------
        dict : Impact metrics including mean change, percentage change, etc.
        """
        if self.trace is None:
            raise ValueError("Model not fitted. Call fit_model() first.")
        
        summary = self.get_change_point_summary()
        change_idx = summary['most_probable_change_point']
        
        before_data = self.data[:change_idx]
        after_data = self.data[change_idx:]
        
        before_mean = np.mean(before_data)
        after_mean = np.mean(after_data)
        before_std = np.std(before_data)
        after_std = np.std(after_data)
        
        mean_change = after_mean - before_mean
        percent_change = (mean_change / before_mean) * 100 if before_mean != 0 else 0
        
        # Statistical significance test
        from scipy import stats
        t_stat, p_value = stats.ttest_ind(before_data, after_data)
        
        impact = {
            'before_mean': before_mean,
            'after_mean': after_mean,
            'mean_change': mean_change,
            'percent_change': percent_change,
            'before_std': before_std,
            'after_std': after_std,
            'std_change': after_std - before_std,
            't_statistic': t_stat,
            'p_value': p_value,
            'significant': p_value < 0.05
        }
        
        return impact

def prepare_data(file_path):
    """
    Prepare data for change point analysis.
    
    Parameters:
    -----------
    file_path : str
        Path to the CSV file containing oil price data
        
    Returns:
    --------
    tuple : (prices, log_returns, dates)
    """
    # Read data
    df = pd.read_csv(file_path)
    
    # Handle different date formats
    try:
        df['Date'] = pd.to_datetime(df['Date'], format='%d-%b-%y')
    except:
        try:
            df['Date'] = pd.to_datetime(df['Date'], format='%b %d, %Y')
        except:
            df['Date'] = pd.to_datetime(df['Date'])
    
    # Sort by date
    df = df.sort_values('Date').reset_index(drop=True)
    
    # Calculate log returns
    prices = df['Price'].values
    log_returns = np.diff(np.log(prices))
    
    return prices, log_returns, df['Date'].values

if __name__ == "__main__":
    # Example usage
    data_path = "../../data/raw/brent_oil_prices.csv"
    
    # Prepare data
    prices, log_returns, dates = prepare_data(data_path)
    
    # Use log returns for analysis (more stationary)
    model = BayesianChangePointModel(log_returns, dates[1:])  # dates[1:] because log_returns is one element shorter
    
    # Build and fit model
    model.build_model(model_type='mean_change')
    trace = model.fit_model(draws=1000, tune=500, chains=2)
    
    # Get results
    change_summary = model.get_change_point_summary()
    param_summary = model.get_parameter_summary()
    impact = model.quantify_impact()
    
    print("Change Point Analysis Results:")
    print(f"Most probable change date: {change_summary['change_date']}")
    print(f"95% Credible interval: {change_summary['credible_interval_lower']} to {change_summary['credible_interval_upper']}")
    print(f"Mean change: {impact['mean_change']:.4f}")
    print(f"Percentage change: {impact['percent_change']:.2f}%")
    print(f"Statistical significance: {'Yes' if impact['significant'] else 'No'} (p={impact['p_value']:.4f})")
    
    # Plot results
    fig = model.plot_results()
    plt.show()