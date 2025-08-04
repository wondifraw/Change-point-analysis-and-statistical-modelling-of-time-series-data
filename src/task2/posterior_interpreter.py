"""
Bayesian Posterior Interpretation Module
Provides detailed interpretation of Bayesian change point results
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import seaborn as sns

@dataclass
class PosteriorSummary:
    """Summary of posterior distributions"""
    change_point_location: int
    posterior_probability: float
    credible_interval_95: Tuple[int, int]
    credible_interval_68: Tuple[int, int]
    uncertainty_width: int
    confidence_level: str

class BayesianPosteriorInterpreter:
    """
    Interprets Bayesian change point detection results
    Provides credible intervals, uncertainty quantification, and practical interpretation
    """
    
    def __init__(self, mcmc_results: Dict):
        """
        Initialize with MCMC results from BayesianChangePointMCMC
        
        Args:
            mcmc_results: Results dictionary from MCMC sampling
        """
        self.mcmc_results = mcmc_results
        self.samples = mcmc_results['samples']
        self.posterior_stats = mcmc_results['posterior_stats']
        
    def interpret_posterior_distributions(self) -> Dict:
        """
        Provide comprehensive interpretation of posterior distributions
        
        Returns:
            Dictionary with detailed interpretations
        """
        interpretation = {
            'change_point_analysis': self._interpret_change_points(),
            'parameter_analysis': self._interpret_parameters(),
            'uncertainty_analysis': self._interpret_uncertainty(),
            'model_diagnostics': self._interpret_model_diagnostics(),
            'practical_implications': self._generate_practical_implications()
        }
        
        return interpretation
    
    def _interpret_change_points(self) -> Dict:
        """Interpret change point posterior distributions"""
        cp_probs = self.posterior_stats['changepoint_probabilities']
        credible_cps = self.posterior_stats['credible_changepoints']
        
        # Create posterior summaries for each credible change point
        summaries = []
        for cp_location, probability in credible_cps.items():
            # Calculate credible intervals
            all_cps = [cp for sample in self.samples for cp in sample['changepoints']]
            nearby_cps = [cp for cp in all_cps if abs(cp - cp_location) <= 10]
            
            if nearby_cps:
                ci_95 = (int(np.percentile(nearby_cps, 2.5)), 
                        int(np.percentile(nearby_cps, 97.5)))
                ci_68 = (int(np.percentile(nearby_cps, 16)), 
                        int(np.percentile(nearby_cps, 84)))
                uncertainty_width = ci_95[1] - ci_95[0]
                
                # Confidence level interpretation
                if probability > 0.8:
                    confidence = "Very High"
                elif probability > 0.6:
                    confidence = "High"
                elif probability > 0.4:
                    confidence = "Moderate"
                else:
                    confidence = "Low"
                
                summary = PosteriorSummary(
                    change_point_location=cp_location,
                    posterior_probability=probability,
                    credible_interval_95=ci_95,
                    credible_interval_68=ci_68,
                    uncertainty_width=uncertainty_width,
                    confidence_level=confidence
                )
                summaries.append(summary)
        
        return {
            'credible_change_points': summaries,
            'total_credible_points': len(summaries),
            'highest_probability_point': max(credible_cps.items(), key=lambda x: x[1]) if credible_cps else None,
            'interpretation': self._generate_cp_interpretation(summaries)
        }
    
    def _interpret_parameters(self) -> Dict:
        """Interpret parameter posterior distributions"""
        mean_estimates = self.posterior_stats.get('mean_estimates', [])
        var_estimates = self.posterior_stats.get('variance_estimates', [])
        
        parameter_interpretation = {
            'regime_means': [],
            'regime_variances': [],
            'regime_comparison': []
        }
        
        for i, (mean_est, var_est) in enumerate(zip(mean_estimates, var_estimates)):
            if mean_est and var_est:
                parameter_interpretation['regime_means'].append({
                    'regime': i + 1,
                    'posterior_mean': mean_est['mean'],
                    'posterior_std': mean_est['std'],
                    'credible_interval': mean_est['credible_interval'],
                    'interpretation': self._interpret_regime_mean(mean_est)
                })
                
                parameter_interpretation['regime_variances'].append({
                    'regime': i + 1,
                    'posterior_mean': var_est['mean'],
                    'posterior_std': var_est['std'],
                    'credible_interval': var_est['credible_interval'],
                    'interpretation': self._interpret_regime_variance(var_est)
                })
        
        # Compare regimes
        if len(mean_estimates) > 1:
            parameter_interpretation['regime_comparison'] = self._compare_regimes(mean_estimates, var_estimates)
        
        return parameter_interpretation
    
    def _interpret_uncertainty(self) -> Dict:
        """Quantify and interpret uncertainty in results"""
        n_cp_dist = self.posterior_stats['n_changepoints_distribution']
        
        # Uncertainty in number of change points
        n_cp_entropy = -sum(p/sum(n_cp_dist.values()) * np.log(p/sum(n_cp_dist.values())) 
                           for p in n_cp_dist.values() if p > 0)
        
        # Uncertainty in change point locations
        cp_probs = self.posterior_stats['changepoint_probabilities']
        location_uncertainty = np.std(list(cp_probs.keys())) if cp_probs else 0
        
        return {
            'number_uncertainty': {
                'entropy': n_cp_entropy,
                'distribution': n_cp_dist,
                'interpretation': self._interpret_number_uncertainty(n_cp_entropy)
            },
            'location_uncertainty': {
                'std_deviation': location_uncertainty,
                'interpretation': self._interpret_location_uncertainty(location_uncertainty)
            },
            'overall_confidence': self._assess_overall_confidence()
        }
    
    def _interpret_model_diagnostics(self) -> Dict:
        """Interpret MCMC diagnostics"""
        acceptance_rate = self.mcmc_results['acceptance_rate']
        n_samples = self.mcmc_results['n_samples']
        
        # Assess convergence
        convergence_assessment = "Good" if 0.2 <= acceptance_rate <= 0.7 else "Poor"
        
        # Effective sample size (simplified)
        eff_sample_size = n_samples * min(acceptance_rate * 2, 1.0)
        
        return {
            'acceptance_rate': acceptance_rate,
            'convergence_assessment': convergence_assessment,
            'effective_sample_size': eff_sample_size,
            'recommendations': self._generate_mcmc_recommendations(acceptance_rate)
        }
    
    def _generate_practical_implications(self) -> List[str]:
        """Generate practical implications of the analysis"""
        implications = []
        
        credible_cps = self.posterior_stats['credible_changepoints']
        most_probable_n = self.posterior_stats['most_probable_n_changepoints']
        
        # Number of change points implication
        if most_probable_n == 0:
            implications.append("📊 The data shows no strong evidence of structural breaks - the time series appears stable.")
        elif most_probable_n == 1:
            implications.append("📍 Evidence suggests a single major regime change in the time series.")
        else:
            implications.append(f"🔄 Evidence suggests {most_probable_n} distinct regime changes, indicating multiple structural shifts.")
        
        # Uncertainty implications
        if len(credible_cps) > most_probable_n:
            implications.append("⚠️ High uncertainty in change point locations - multiple plausible scenarios exist.")
        
        # Confidence implications
        high_conf_points = [cp for cp, prob in credible_cps.items() if prob > 0.7]
        if high_conf_points:
            implications.append(f"✅ {len(high_conf_points)} change point(s) have high posterior confidence (>70%).")
        
        return implications
    
    def plot_posterior_analysis(self, data: Optional[pd.DataFrame] = None, 
                              save_path: Optional[str] = None) -> None:
        """Create comprehensive posterior analysis plots"""
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('Bayesian Posterior Analysis', fontsize=16, fontweight='bold')
        
        # Plot 1: Number of change points distribution
        ax1 = axes[0, 0]
        n_cp_dist = self.posterior_stats['n_changepoints_distribution']
        ax1.bar(n_cp_dist.keys(), n_cp_dist.values(), alpha=0.7, color='skyblue')
        ax1.set_xlabel('Number of Change Points')
        ax1.set_ylabel('Posterior Probability')
        ax1.set_title('Posterior: Number of Change Points')
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Change point locations with credible intervals
        ax2 = axes[0, 1]
        cp_probs = self.posterior_stats['changepoint_probabilities']
        if cp_probs:
            locations = list(cp_probs.keys())
            probabilities = list(cp_probs.values())
            
            bars = ax2.bar(locations, probabilities, alpha=0.7, color='orange')
            ax2.set_xlabel('Time Index')
            ax2.set_ylabel('Posterior Probability')
            ax2.set_title('Change Point Location Probabilities')
            ax2.grid(True, alpha=0.3)
            
            # Add credible intervals
            credible_cps = self.posterior_stats['credible_changepoints']
            for cp, prob in credible_cps.items():
                if prob > 0.3:  # Only show high-probability points
                    # Simplified credible interval
                    ci_width = 20  # Approximate
                    ax2.errorbar(cp, prob, xerr=ci_width, fmt='ro', capsize=5, alpha=0.7)
        
        # Plot 3: MCMC trace for number of change points
        ax3 = axes[0, 2]
        n_cp_trace = [sample['n_changepoints'] for sample in self.samples]
        ax3.plot(n_cp_trace, alpha=0.7)
        ax3.set_xlabel('MCMC Iteration')
        ax3.set_ylabel('Number of Change Points')
        ax3.set_title('MCMC Trace: # Change Points')
        ax3.grid(True, alpha=0.3)
        
        # Plot 4: Posterior predictive check (if data provided)
        ax4 = axes[1, 0]
        if data is not None:
            ax4.plot(data.index, data['price'], 'b-', alpha=0.7, label='Observed Data')
            
            # Add credible change points with uncertainty bands
            credible_cps = self.posterior_stats['credible_changepoints']
            for cp, prob in credible_cps.items():
                if prob > 0.3:
                    ax4.axvline(cp, color='red', alpha=prob, linestyle='--', 
                              linewidth=2, label=f'CP (p={prob:.2f})')
                    
                    # Add uncertainty band
                    ci_width = 20
                    ax4.axvspan(cp - ci_width, cp + ci_width, alpha=0.1, color='red')
            
            ax4.set_xlabel('Time')
            ax4.set_ylabel('Price')
            ax4.set_title('Data with Posterior Change Points')
            ax4.legend()
        else:
            ax4.text(0.5, 0.5, 'Data not provided\nfor visualization', 
                    ha='center', va='center', transform=ax4.transAxes)
            ax4.set_title('Posterior Predictive Check')
        
        # Plot 5: Uncertainty quantification
        ax5 = axes[1, 1]
        
        # Create uncertainty heatmap
        if cp_probs:
            # Simplified uncertainty visualization
            locations = np.array(list(cp_probs.keys()))
            probs = np.array(list(cp_probs.values()))
            
            # Create 2D grid for heatmap
            x_grid = np.linspace(min(locations), max(locations), 50)
            uncertainty = np.zeros_like(x_grid)
            
            for i, x in enumerate(x_grid):
                # Calculate uncertainty as inverse of max probability in neighborhood
                nearby_probs = [p for loc, p in zip(locations, probs) if abs(loc - x) <= 10]
                uncertainty[i] = 1 - max(nearby_probs) if nearby_probs else 1
            
            ax5.plot(x_grid, uncertainty, 'r-', linewidth=2)
            ax5.fill_between(x_grid, uncertainty, alpha=0.3, color='red')
            ax5.set_xlabel('Time Index')
            ax5.set_ylabel('Uncertainty (1 - max prob)')
            ax5.set_title('Change Point Location Uncertainty')
            ax5.grid(True, alpha=0.3)
        
        # Plot 6: Model comparison metrics
        ax6 = axes[1, 2]
        
        # Create summary metrics visualization
        metrics = {
            'Acceptance Rate': self.mcmc_results['acceptance_rate'],
            'Avg # Change Points': np.mean([s['n_changepoints'] for s in self.samples]),
            'Max Probability': max(cp_probs.values()) if cp_probs else 0,
            'Uncertainty': np.std([s['n_changepoints'] for s in self.samples])
        }
        
        bars = ax6.bar(range(len(metrics)), list(metrics.values()), 
                      color=['green', 'blue', 'orange', 'red'], alpha=0.7)
        ax6.set_xticks(range(len(metrics)))
        ax6.set_xticklabels(list(metrics.keys()), rotation=45, ha='right')
        ax6.set_ylabel('Value')
        ax6.set_title('Model Performance Metrics')
        ax6.grid(True, alpha=0.3)
        
        # Add value labels on bars
        for bar, value in zip(bars, metrics.values()):
            height = bar.get_height()
            ax6.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{value:.3f}', ha='center', va='bottom')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def generate_interpretation_report(self) -> str:
        """Generate comprehensive interpretation report"""
        interpretation = self.interpret_posterior_distributions()
        
        report = []
        report.append("=" * 80)
        report.append("BAYESIAN POSTERIOR INTERPRETATION REPORT")
        report.append("=" * 80)
        
        # Change point analysis
        cp_analysis = interpretation['change_point_analysis']
        report.append(f"\n🎯 CHANGE POINT ANALYSIS")
        report.append("-" * 40)
        report.append(f"Credible Change Points Detected: {cp_analysis['total_credible_points']}")
        
        if cp_analysis['highest_probability_point']:
            cp_loc, cp_prob = cp_analysis['highest_probability_point']
            report.append(f"Highest Probability Point: Index {cp_loc} (p = {cp_prob:.3f})")
        
        for summary in cp_analysis['credible_change_points']:
            report.append(f"\n📍 Change Point at Index {summary.change_point_location}:")
            report.append(f"  Posterior Probability: {summary.posterior_probability:.3f}")
            report.append(f"  95% Credible Interval: [{summary.credible_interval_95[0]}, {summary.credible_interval_95[1]}]")
            report.append(f"  68% Credible Interval: [{summary.credible_interval_68[0]}, {summary.credible_interval_68[1]}]")
            report.append(f"  Uncertainty Width: ±{summary.uncertainty_width//2} time units")
            report.append(f"  Confidence Level: {summary.confidence_level}")
        
        # Uncertainty analysis
        uncertainty = interpretation['uncertainty_analysis']
        report.append(f"\n🔍 UNCERTAINTY ANALYSIS")
        report.append("-" * 40)
        report.append(f"Number Uncertainty: {uncertainty['number_uncertainty']['interpretation']}")
        report.append(f"Location Uncertainty: {uncertainty['location_uncertainty']['interpretation']}")
        report.append(f"Overall Confidence: {uncertainty['overall_confidence']}")
        
        # Model diagnostics
        diagnostics = interpretation['model_diagnostics']
        report.append(f"\n⚙️ MODEL DIAGNOSTICS")
        report.append("-" * 40)
        report.append(f"MCMC Acceptance Rate: {diagnostics['acceptance_rate']:.3f}")
        report.append(f"Convergence Assessment: {diagnostics['convergence_assessment']}")
        report.append(f"Effective Sample Size: {diagnostics['effective_sample_size']:.0f}")
        
        # Practical implications
        implications = interpretation['practical_implications']
        report.append(f"\n💡 PRACTICAL IMPLICATIONS")
        report.append("-" * 40)
        for implication in implications:
            report.append(f"  {implication}")
        
        return "\n".join(report)
    
    # Helper methods for interpretations
    def _generate_cp_interpretation(self, summaries: List[PosteriorSummary]) -> str:
        if not summaries:
            return "No credible change points detected with sufficient confidence."
        
        high_conf = [s for s in summaries if s.confidence_level in ["High", "Very High"]]
        if high_conf:
            return f"{len(high_conf)} change point(s) detected with high confidence."
        else:
            return f"{len(summaries)} change point(s) detected with moderate confidence."
    
    def _interpret_regime_mean(self, mean_est: Dict) -> str:
        ci_width = mean_est['credible_interval'][1] - mean_est['credible_interval'][0]
        if ci_width < mean_est['mean'] * 0.1:
            return "Well-estimated (narrow credible interval)"
        else:
            return "Uncertain estimate (wide credible interval)"
    
    def _interpret_regime_variance(self, var_est: Dict) -> str:
        if var_est['mean'] > var_est['std'] * 3:
            return "Stable variance estimate"
        else:
            return "Uncertain variance estimate"
    
    def _compare_regimes(self, mean_estimates: List, var_estimates: List) -> List[str]:
        comparisons = []
        for i in range(len(mean_estimates) - 1):
            mean1, mean2 = mean_estimates[i]['mean'], mean_estimates[i+1]['mean']
            change_pct = ((mean2 - mean1) / mean1) * 100
            comparisons.append(f"Regime {i+1} → {i+2}: {change_pct:+.1f}% change in mean")
        return comparisons
    
    def _interpret_number_uncertainty(self, entropy: float) -> str:
        if entropy < 0.5:
            return "Low uncertainty - clear evidence for specific number of change points"
        elif entropy < 1.0:
            return "Moderate uncertainty - some ambiguity in number of change points"
        else:
            return "High uncertainty - multiple scenarios equally plausible"
    
    def _interpret_location_uncertainty(self, std_dev: float) -> str:
        if std_dev < 10:
            return "Precise location estimates"
        elif std_dev < 30:
            return "Moderate location uncertainty"
        else:
            return "High location uncertainty"
    
    def _assess_overall_confidence(self) -> str:
        credible_cps = self.posterior_stats['credible_changepoints']
        if not credible_cps:
            return "Low - no credible change points"
        
        avg_prob = np.mean(list(credible_cps.values()))
        if avg_prob > 0.7:
            return "High"
        elif avg_prob > 0.5:
            return "Moderate"
        else:
            return "Low"
    
    def _generate_mcmc_recommendations(self, acceptance_rate: float) -> List[str]:
        recommendations = []
        if acceptance_rate < 0.2:
            recommendations.append("Consider reducing step size or proposal variance")
        elif acceptance_rate > 0.7:
            recommendations.append("Consider increasing step size or proposal variance")
        else:
            recommendations.append("MCMC tuning appears adequate")
        
        recommendations.append("Consider running longer chains for better convergence")
        return recommendations