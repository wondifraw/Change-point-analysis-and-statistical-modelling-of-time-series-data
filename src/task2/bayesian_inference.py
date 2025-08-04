"""
Advanced Bayesian Change Point Detection with MCMC Sampling
Provides posterior distributions and uncertainty quantification
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional
import logging

class BayesianChangePointMCMC:
    """
    Bayesian change point detection using MCMC sampling
    Provides full posterior distributions without PyMC3 dependency
    """
    
    def __init__(self, data: np.ndarray, max_changepoints: int = 3):
        self.data = data
        self.n = len(data)
        self.max_changepoints = max_changepoints
        self.samples = []
        self.posterior_stats = {}
        
    def log_likelihood(self, changepoints: List[int], means: List[float], variances: List[float]) -> float:
        """Calculate log likelihood for given parameters"""
        if not changepoints:
            changepoints = []
        
        # Add boundaries
        boundaries = [0] + sorted(changepoints) + [self.n]
        
        if len(means) != len(boundaries) - 1 or len(variances) != len(boundaries) - 1:
            return -np.inf
        
        log_lik = 0.0
        for i in range(len(boundaries) - 1):
            start, end = boundaries[i], boundaries[i + 1]
            if end <= start:
                return -np.inf
                
            segment = self.data[start:end]
            if len(segment) == 0:
                return -np.inf
                
            # Gaussian likelihood
            log_lik += -0.5 * len(segment) * np.log(2 * np.pi * variances[i])
            log_lik += -0.5 * np.sum((segment - means[i])**2) / variances[i]
        
        return log_lik
    
    def log_prior(self, changepoints: List[int], means: List[float], variances: List[float]) -> float:
        """Calculate log prior probability"""
        # Prior on number of changepoints (Poisson)
        lambda_cp = 1.0
        log_prior_cp = len(changepoints) * np.log(lambda_cp) - lambda_cp
        
        # Prior on changepoint locations (uniform)
        log_prior_loc = -len(changepoints) * np.log(self.n) if changepoints else 0
        
        # Prior on means (normal)
        data_mean = np.mean(self.data)
        data_std = np.std(self.data)
        log_prior_means = -0.5 * np.sum([(m - data_mean)**2 / (data_std**2) for m in means])
        
        # Prior on variances (inverse gamma)
        alpha, beta = 2.0, data_std**2
        log_prior_vars = np.sum([-(alpha + 1) * np.log(v) - beta/v for v in variances])
        
        return log_prior_cp + log_prior_loc + log_prior_means + log_prior_vars
    
    def propose_changepoints(self, current_changepoints: List[int]) -> List[int]:
        """Propose new changepoint configuration"""
        proposal = current_changepoints.copy()
        
        if len(proposal) == 0 or (len(proposal) < self.max_changepoints and np.random.rand() < 0.5):
            # Add changepoint
            new_cp = np.random.randint(10, self.n - 10)
            if new_cp not in proposal:
                proposal.append(new_cp)
        elif len(proposal) > 0 and np.random.rand() < 0.3:
            # Remove changepoint
            if proposal:
                proposal.remove(np.random.choice(proposal))
        else:
            # Move changepoint
            if proposal:
                idx = np.random.randint(len(proposal))
                proposal[idx] = max(10, min(self.n - 10, 
                                          proposal[idx] + np.random.randint(-20, 21)))
        
        return sorted(list(set(proposal)))
    
    def propose_parameters(self, changepoints: List[int], current_means: List[float], 
                          current_vars: List[float]) -> Tuple[List[float], List[float]]:
        """Propose new mean and variance parameters"""
        n_segments = len(changepoints) + 1
        
        # Calculate segment statistics for informed proposals
        boundaries = [0] + sorted(changepoints) + [self.n]
        new_means = []
        new_vars = []
        
        for i in range(n_segments):
            start, end = boundaries[i], boundaries[i + 1]
            segment = self.data[start:end]
            
            if len(segment) > 0:
                seg_mean = np.mean(segment)
                seg_var = np.var(segment) + 1e-6  # Add small constant for stability
                
                # Propose around segment statistics
                new_mean = np.random.normal(seg_mean, np.sqrt(seg_var) / np.sqrt(len(segment)))
                new_var = seg_var * np.random.gamma(len(segment)/2, 2/len(segment))
                
                new_means.append(new_mean)
                new_vars.append(max(1e-6, new_var))
            else:
                new_means.append(np.mean(self.data))
                new_vars.append(np.var(self.data))
        
        return new_means, new_vars
    
    def mcmc_sample(self, n_samples: int = 5000, burn_in: int = 1000) -> Dict:
        """
        Run MCMC sampling to get posterior distributions
        
        Args:
            n_samples: Number of MCMC samples
            burn_in: Number of burn-in samples to discard
            
        Returns:
            Dictionary with samples and posterior statistics
        """
        # Initialize
        current_changepoints = []
        current_means = [np.mean(self.data)]
        current_vars = [np.var(self.data)]
        
        current_log_posterior = (self.log_likelihood(current_changepoints, current_means, current_vars) + 
                                self.log_prior(current_changepoints, current_means, current_vars))
        
        samples = []
        accepted = 0
        
        for i in range(n_samples + burn_in):
            # Propose new state
            if np.random.rand() < 0.7:  # Update changepoints
                proposed_changepoints = self.propose_changepoints(current_changepoints)
                proposed_means, proposed_vars = self.propose_parameters(
                    proposed_changepoints, current_means, current_vars)
            else:  # Update parameters only
                proposed_changepoints = current_changepoints.copy()
                proposed_means, proposed_vars = self.propose_parameters(
                    proposed_changepoints, current_means, current_vars)
            
            # Calculate acceptance probability
            proposed_log_posterior = (self.log_likelihood(proposed_changepoints, proposed_means, proposed_vars) + 
                                    self.log_prior(proposed_changepoints, proposed_means, proposed_vars))
            
            log_alpha = min(0, proposed_log_posterior - current_log_posterior)
            
            # Accept or reject
            if np.log(np.random.rand()) < log_alpha:
                current_changepoints = proposed_changepoints
                current_means = proposed_means
                current_vars = proposed_vars
                current_log_posterior = proposed_log_posterior
                accepted += 1
            
            # Store sample (after burn-in)
            if i >= burn_in:
                samples.append({
                    'changepoints': current_changepoints.copy(),
                    'means': current_means.copy(),
                    'variances': current_vars.copy(),
                    'log_posterior': current_log_posterior,
                    'n_changepoints': len(current_changepoints)
                })
        
        self.samples = samples
        acceptance_rate = accepted / (n_samples + burn_in)
        
        # Calculate posterior statistics
        self.posterior_stats = self._calculate_posterior_stats()
        
        return {
            'samples': samples,
            'acceptance_rate': acceptance_rate,
            'posterior_stats': self.posterior_stats,
            'n_samples': n_samples,
            'burn_in': burn_in
        }
    
    def _calculate_posterior_stats(self) -> Dict:
        """Calculate posterior statistics from samples"""
        if not self.samples:
            return {}
        
        # Number of changepoints distribution
        n_cp_counts = {}
        all_changepoints = []
        
        for sample in self.samples:
            n_cp = sample['n_changepoints']
            n_cp_counts[n_cp] = n_cp_counts.get(n_cp, 0) + 1
            all_changepoints.extend(sample['changepoints'])
        
        # Most probable number of changepoints
        most_probable_n_cp = max(n_cp_counts.items(), key=lambda x: x[1])[0]
        
        # Changepoint location probabilities
        cp_probs = {}
        for cp in all_changepoints:
            # Group nearby changepoints (within 5 time units)
            found_group = False
            for existing_cp in cp_probs:
                if abs(cp - existing_cp) <= 5:
                    cp_probs[existing_cp] += 1
                    found_group = True
                    break
            if not found_group:
                cp_probs[cp] = 1
        
        # Normalize probabilities
        total_samples = len(self.samples)
        cp_probs = {cp: count/total_samples for cp, count in cp_probs.items()}
        
        # Credible changepoints (probability > 0.1)
        credible_changepoints = {cp: prob for cp, prob in cp_probs.items() if prob > 0.1}
        
        # Parameter estimates for most probable configuration
        samples_with_most_probable_n_cp = [s for s in self.samples if s['n_changepoints'] == most_probable_n_cp]
        
        if samples_with_most_probable_n_cp:
            mean_estimates = []
            var_estimates = []
            
            # Average over samples with same number of changepoints
            n_segments = most_probable_n_cp + 1
            for seg in range(n_segments):
                seg_means = [s['means'][seg] if seg < len(s['means']) else np.nan 
                           for s in samples_with_most_probable_n_cp]
                seg_vars = [s['variances'][seg] if seg < len(s['variances']) else np.nan 
                          for s in samples_with_most_probable_n_cp]
                
                mean_estimates.append({
                    'mean': np.nanmean(seg_means),
                    'std': np.nanstd(seg_means),
                    'credible_interval': np.nanpercentile(seg_means, [2.5, 97.5])
                })
                
                var_estimates.append({
                    'mean': np.nanmean(seg_vars),
                    'std': np.nanstd(seg_vars),
                    'credible_interval': np.nanpercentile(seg_vars, [2.5, 97.5])
                })
        else:
            mean_estimates = []
            var_estimates = []
        
        return {
            'n_changepoints_distribution': n_cp_counts,
            'most_probable_n_changepoints': most_probable_n_cp,
            'changepoint_probabilities': cp_probs,
            'credible_changepoints': credible_changepoints,
            'mean_estimates': mean_estimates,
            'variance_estimates': var_estimates,
            'total_samples': total_samples
        }
    
    def plot_posterior_results(self, save_path: Optional[str] = None) -> None:
        """Plot posterior distributions and results"""
        if not self.samples:
            print("No samples available. Run mcmc_sample() first.")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # Plot 1: Number of changepoints distribution
        n_cp_dist = self.posterior_stats['n_changepoints_distribution']
        axes[0, 0].bar(n_cp_dist.keys(), n_cp_dist.values())
        axes[0, 0].set_xlabel('Number of Change Points')
        axes[0, 0].set_ylabel('Posterior Probability')
        axes[0, 0].set_title('Posterior Distribution of Number of Change Points')
        
        # Plot 2: Changepoint locations
        cp_probs = self.posterior_stats['changepoint_probabilities']
        if cp_probs:
            axes[0, 1].bar(cp_probs.keys(), cp_probs.values())
            axes[0, 1].set_xlabel('Time Index')
            axes[0, 1].set_ylabel('Posterior Probability')
            axes[0, 1].set_title('Change Point Location Probabilities')
        
        # Plot 3: Data with credible changepoints
        axes[1, 0].plot(self.data, 'b-', alpha=0.7, label='Data')
        credible_cps = self.posterior_stats['credible_changepoints']
        for cp, prob in credible_cps.items():
            axes[1, 0].axvline(cp, color='red', alpha=prob, 
                              label=f'CP at {cp} (p={prob:.2f})')
        axes[1, 0].set_xlabel('Time')
        axes[1, 0].set_ylabel('Value')
        axes[1, 0].set_title('Data with Credible Change Points')
        axes[1, 0].legend()
        
        # Plot 4: MCMC trace for number of changepoints
        n_cp_trace = [s['n_changepoints'] for s in self.samples]
        axes[1, 1].plot(n_cp_trace)
        axes[1, 1].set_xlabel('MCMC Iteration')
        axes[1, 1].set_ylabel('Number of Change Points')
        axes[1, 1].set_title('MCMC Trace: Number of Change Points')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def get_point_estimates(self) -> Dict:
        """Get point estimates and credible intervals"""
        if not self.posterior_stats:
            return {}
        
        credible_cps = self.posterior_stats['credible_changepoints']
        most_probable_cps = sorted([cp for cp, prob in credible_cps.items() if prob > 0.3])
        
        return {
            'most_probable_changepoints': most_probable_cps,
            'changepoint_probabilities': credible_cps,
            'most_probable_n_changepoints': self.posterior_stats['most_probable_n_changepoints'],
            'parameter_estimates': {
                'means': self.posterior_stats['mean_estimates'],
                'variances': self.posterior_stats['variance_estimates']
            }
        }