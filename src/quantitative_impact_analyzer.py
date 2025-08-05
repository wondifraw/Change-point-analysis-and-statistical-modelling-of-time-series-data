"""
Quantitative Impact Analysis Module
Provides detailed statistical analysis of change point impacts
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import scipy.stats as stats

@dataclass
class ImpactMetrics:
    """Container for quantitative impact metrics"""
    change_point_date: str
    change_point_index: int
    
    # Mean changes
    mean_before: float
    mean_after: float
    mean_change_absolute: float
    mean_change_percent: float
    
    # Variance changes
    variance_before: float
    variance_after: float
    variance_change_percent: float
    
    # Volatility changes
    volatility_before: float
    volatility_after: float
    volatility_change_percent: float
    
    # Statistical significance
    t_statistic: float
    p_value: float
    effect_size_cohens_d: float
    
    # Confidence intervals
    mean_diff_ci_lower: float
    mean_diff_ci_upper: float
    confidence_level: float = 0.95

class QuantitativeImpactAnalyzer:
    """
    Analyzes quantitative impact of change points on time series
    """
    
    def __init__(self, data: pd.DataFrame, change_points: List[int]):
        """
        Initialize analyzer with data and detected change points
        
        Args:
            data: DataFrame with 'date' and 'price' columns
            change_points: List of change point indices
        """
        self.data = data
        self.change_points = sorted(change_points)
        self.prices = data['price'].values
        
    def analyze_all_impacts(self) -> List[ImpactMetrics]:
        """Analyze impact for all change points"""
        impacts = []
        
        for cp_idx in self.change_points:
            impact = self._analyze_single_impact(cp_idx)
            impacts.append(impact)
            
        return impacts
    
    def _analyze_single_impact(self, cp_idx: int) -> ImpactMetrics:
        """Analyze quantitative impact of a single change point with Bayesian credible intervals"""
        # Define temporal segments around change point
        before_segment = self.prices[:cp_idx]
        after_segment = self.prices[cp_idx:]
        
        # Calculate fundamental statistical moments
        mean_before = np.mean(before_segment)
        mean_after = np.mean(after_segment)
        var_before = np.var(before_segment, ddof=1)
        var_after = np.var(after_segment, ddof=1)
        
        # Bayesian posterior credible intervals for change point location
        posterior_ci = self._calculate_posterior_credible_interval(cp_idx)
        
        # Effect size measures using multiple metrics
        effect_sizes = self._calculate_comprehensive_effect_sizes(before_segment, after_segment)
        
        # Calculate changes
        mean_change_abs = mean_after - mean_before
        mean_change_pct = (mean_change_abs / mean_before) * 100 if mean_before != 0 else 0
        var_change_pct = ((var_after - var_before) / var_before) * 100 if var_before != 0 else 0
        
        # Volatility (using log returns)
        vol_before = self._calculate_volatility(before_segment)
        vol_after = self._calculate_volatility(after_segment)
        vol_change_pct = ((vol_after - vol_before) / vol_before) * 100 if vol_before != 0 else 0
        
        # Statistical tests
        t_stat, p_val = stats.ttest_ind(before_segment, after_segment, equal_var=False)
        
        # Effect size (Cohen's d)
        pooled_std = np.sqrt(((len(before_segment) - 1) * var_before + 
                             (len(after_segment) - 1) * var_after) / 
                            (len(before_segment) + len(after_segment) - 2))
        cohens_d = mean_change_abs / pooled_std if pooled_std != 0 else 0
        
        # Confidence interval for mean difference
        se_diff = np.sqrt(var_before/len(before_segment) + var_after/len(after_segment))
        df = len(before_segment) + len(after_segment) - 2
        t_critical = stats.t.ppf(0.975, df)  # 95% CI
        ci_lower = mean_change_abs - t_critical * se_diff
        ci_upper = mean_change_abs + t_critical * se_diff
        
        return ImpactMetrics(
            change_point_date=self.data.iloc[cp_idx]['date'].strftime('%Y-%m-%d'),
            change_point_index=cp_idx,
            mean_before=mean_before,
            mean_after=mean_after,
            mean_change_absolute=mean_change_abs,
            mean_change_percent=mean_change_pct,
            variance_before=var_before,
            variance_after=var_after,
            variance_change_percent=var_change_pct,
            volatility_before=vol_before,
            volatility_after=vol_after,
            volatility_change_percent=vol_change_pct,
            t_statistic=t_stat,
            p_value=p_val,
            effect_size_cohens_d=cohens_d,
            mean_diff_ci_lower=ci_lower,
            mean_diff_ci_upper=ci_upper,
            posterior_credible_interval=posterior_ci,
            comprehensive_effect_sizes=effect_sizes
        )
    
    def _calculate_posterior_credible_interval(self, cp_idx: int, confidence: float = 0.95) -> Tuple[int, int]:
        """Calculate Bayesian posterior credible interval for change point location"""
        # Simplified Bayesian approach using likelihood-based confidence region
        window_size = min(50, len(self.prices) // 10)  # Adaptive window size
        
        # Calculate likelihood profile around change point
        likelihood_profile = []
        test_points = range(max(0, cp_idx - window_size), 
                          min(len(self.prices), cp_idx + window_size + 1))
        
        for test_cp in test_points:
            if test_cp <= 10 or test_cp >= len(self.prices) - 10:
                likelihood_profile.append(-np.inf)
                continue
                
            # Calculate log-likelihood for this change point location
            before_seg = self.prices[:test_cp]
            after_seg = self.prices[test_cp:]
            
            if len(before_seg) > 0 and len(after_seg) > 0:
                # Gaussian likelihood assuming different means and variances
                ll_before = -0.5 * len(before_seg) * np.log(2 * np.pi * np.var(before_seg))
                ll_before -= 0.5 * np.sum((before_seg - np.mean(before_seg))**2) / np.var(before_seg)
                
                ll_after = -0.5 * len(after_seg) * np.log(2 * np.pi * np.var(after_seg))
                ll_after -= 0.5 * np.sum((after_seg - np.mean(after_seg))**2) / np.var(after_seg)
                
                likelihood_profile.append(ll_before + ll_after)
            else:
                likelihood_profile.append(-np.inf)
        
        # Convert to posterior probabilities (uniform prior)
        log_probs = np.array(likelihood_profile)
        log_probs = log_probs - np.max(log_probs)  # Numerical stability
        probs = np.exp(log_probs)
        probs = probs / np.sum(probs)
        
        # Calculate credible interval
        cumsum = np.cumsum(probs)
        alpha = 1 - confidence
        lower_idx = np.searchsorted(cumsum, alpha/2)
        upper_idx = np.searchsorted(cumsum, 1 - alpha/2)
        
        lower_bound = test_points[min(lower_idx, len(test_points)-1)]
        upper_bound = test_points[min(upper_idx, len(test_points)-1)]
        
        return (lower_bound, upper_bound)
    
    def _calculate_comprehensive_effect_sizes(self, before_seg: np.ndarray, after_seg: np.ndarray) -> Dict[str, float]:
        """Calculate multiple effect size measures for comprehensive impact assessment"""
        effect_sizes = {}
        
        # Cohen's d (standardized mean difference)
        pooled_std = np.sqrt(((len(before_seg) - 1) * np.var(before_seg, ddof=1) + 
                             (len(after_seg) - 1) * np.var(after_seg, ddof=1)) / 
                            (len(before_seg) + len(after_seg) - 2))
        
        if pooled_std > 0:
            effect_sizes['cohens_d'] = (np.mean(after_seg) - np.mean(before_seg)) / pooled_std
        else:
            effect_sizes['cohens_d'] = 0.0
        
        # Glass's delta (uses control group standard deviation)
        if np.std(before_seg) > 0:
            effect_sizes['glass_delta'] = (np.mean(after_seg) - np.mean(before_seg)) / np.std(before_seg)
        else:
            effect_sizes['glass_delta'] = 0.0
        
        # Hedges' g (bias-corrected Cohen's d)
        correction_factor = 1 - (3 / (4 * (len(before_seg) + len(after_seg)) - 9))
        effect_sizes['hedges_g'] = effect_sizes['cohens_d'] * correction_factor
        
        # Common Language Effect Size (probability of superiority)
        if len(before_seg) > 0 and len(after_seg) > 0:
            comparisons = np.array([after_val > before_val 
                                  for after_val in after_seg 
                                  for before_val in before_seg])
            effect_sizes['cles'] = np.mean(comparisons)
        else:
            effect_sizes['cles'] = 0.5
        
        # Variance ratio (F-statistic based)
        var_before = np.var(before_seg, ddof=1)
        var_after = np.var(after_seg, ddof=1)
        if var_before > 0:
            effect_sizes['variance_ratio'] = var_after / var_before
        else:
            effect_sizes['variance_ratio'] = 1.0
        
        return effect_sizes
    
    def perform_systematic_event_alignment_test(self, events_df: pd.DataFrame, 
                                              tolerance_days: int = 30) -> Dict:
        """Systematic statistical test for change point-event alignment"""
        alignment_results = {
            'total_change_points': len(self.change_points),
            'aligned_points': 0,
            'alignment_statistics': [],
            'permutation_test_p_value': None,
            'systematic_bias': None
        }
        
        if len(self.change_points) == 0 or len(events_df) == 0:
            return alignment_results
        
        # Calculate actual alignments
        actual_alignments = []
        for cp_idx in self.change_points:
            cp_date = self.data.iloc[cp_idx]['date']
            
            # Find closest event within tolerance
            time_diffs = abs((events_df['date'] - cp_date).dt.days)
            min_diff = time_diffs.min()
            
            if min_diff <= tolerance_days:
                actual_alignments.append(min_diff)
                alignment_results['aligned_points'] += 1
        
        # Permutation test for statistical significance
        if len(actual_alignments) > 0:
            alignment_results['permutation_test_p_value'] = self._permutation_test_alignment(
                actual_alignments, events_df, tolerance_days
            )
        
        # Test for systematic temporal bias
        if len(actual_alignments) >= 3:
            # Calculate if change points systematically occur before/after events
            signed_diffs = []
            for cp_idx in self.change_points:
                cp_date = self.data.iloc[cp_idx]['date']
                time_diffs = (events_df['date'] - cp_date).dt.days
                closest_event_idx = time_diffs.abs().idxmin()
                signed_diffs.append(time_diffs.iloc[closest_event_idx])
            
            # One-sample t-test against zero (no systematic bias)
            from scipy import stats
            t_stat, p_val = stats.ttest_1samp(signed_diffs, 0)
            alignment_results['systematic_bias'] = {
                'mean_offset_days': np.mean(signed_diffs),
                't_statistic': t_stat,
                'p_value': p_val,
                'interpretation': 'Change points occur systematically ' + 
                               ('after' if np.mean(signed_diffs) > 0 else 'before') + 
                               ' events' if p_val < 0.05 else 'No systematic bias detected'
            }
        
        return alignment_results
    
    def _permutation_test_alignment(self, actual_alignments: List[float], 
                                  events_df: pd.DataFrame, tolerance_days: int, 
                                  n_permutations: int = 1000) -> float:
        """Permutation test to assess statistical significance of event alignment"""
        actual_alignment_rate = len(actual_alignments) / len(self.change_points)
        
        # Generate null distribution by randomly permuting change point dates
        null_alignment_rates = []
        
        for _ in range(n_permutations):
            # Randomly shuffle change point indices
            random_cp_indices = np.random.choice(len(self.data), len(self.change_points), replace=False)
            
            # Count alignments for random change points
            random_alignments = 0
            for cp_idx in random_cp_indices:
                cp_date = self.data.iloc[cp_idx]['date']
                time_diffs = abs((events_df['date'] - cp_date).dt.days)
                if time_diffs.min() <= tolerance_days:
                    random_alignments += 1
            
            null_alignment_rates.append(random_alignments / len(self.change_points))
        
        # Calculate p-value
        p_value = np.mean(np.array(null_alignment_rates) >= actual_alignment_rate)
        return p_value
    
    def _calculate_volatility(self, prices: np.ndarray) -> float:
        """Calculate annualized volatility from price series"""
        if len(prices) < 2:
            return 0.0
        
        log_returns = np.diff(np.log(prices + 1e-8))  # Add small constant for stability
        return np.std(log_returns) * np.sqrt(252)  # Annualized (252 trading days)
    
    def generate_impact_report(self, impacts: List[ImpactMetrics]) -> str:
        """Generate comprehensive impact report"""
        report = []
        report.append("=" * 80)
        report.append("QUANTITATIVE CHANGE POINT IMPACT ANALYSIS")
        report.append("=" * 80)
        
        for i, impact in enumerate(impacts, 1):
            report.append(f"\n📍 CHANGE POINT {i}: {impact.change_point_date}")
            report.append("-" * 50)
            
            # Mean impact
            direction = "↗️ INCREASE" if impact.mean_change_percent > 0 else "↘️ DECREASE"
            report.append(f"Mean Price Impact: {direction}")
            report.append(f"  Before: ${impact.mean_before:.2f}")
            report.append(f"  After:  ${impact.mean_after:.2f}")
            report.append(f"  Change: ${impact.mean_change_absolute:+.2f} ({impact.mean_change_percent:+.1f}%)")
            report.append(f"  95% CI: [${impact.mean_diff_ci_lower:.2f}, ${impact.mean_diff_ci_upper:.2f}]")
            
            # Volatility impact
            vol_direction = "↗️ MORE VOLATILE" if impact.volatility_change_percent > 0 else "↘️ LESS VOLATILE"
            report.append(f"\nVolatility Impact: {vol_direction}")
            report.append(f"  Before: {impact.volatility_before:.1%}")
            report.append(f"  After:  {impact.volatility_after:.1%}")
            report.append(f"  Change: {impact.volatility_change_percent:+.1f}%")
            
            # Statistical significance
            significance = "SIGNIFICANT" if impact.p_value < 0.05 else "NOT SIGNIFICANT"
            effect_size = self._interpret_effect_size(impact.effect_size_cohens_d)
            report.append(f"\nStatistical Analysis:")
            report.append(f"  Significance: {significance} (p = {impact.p_value:.4f})")
            report.append(f"  Effect Size: {effect_size} (Cohen's d = {impact.effect_size_cohens_d:.2f})")
            
            # Practical interpretation
            report.append(f"\n💡 Practical Impact:")
            if abs(impact.mean_change_percent) > 20:
                report.append(f"  🔥 MAJOR structural shift ({impact.mean_change_percent:+.1f}%)")
            elif abs(impact.mean_change_percent) > 10:
                report.append(f"  ⚠️ MODERATE regime change ({impact.mean_change_percent:+.1f}%)")
            else:
                report.append(f"  📊 MINOR adjustment ({impact.mean_change_percent:+.1f}%)")
        
        # Summary statistics
        report.append(f"\n" + "=" * 80)
        report.append("SUMMARY STATISTICS")
        report.append("=" * 80)
        
        total_changes = [abs(imp.mean_change_percent) for imp in impacts]
        significant_changes = [imp for imp in impacts if imp.p_value < 0.05]
        
        report.append(f"Total Change Points Analyzed: {len(impacts)}")
        report.append(f"Statistically Significant: {len(significant_changes)} ({len(significant_changes)/len(impacts)*100:.1f}%)")
        report.append(f"Average Impact Magnitude: {np.mean(total_changes):.1f}%")
        report.append(f"Largest Impact: {max(total_changes):.1f}%")
        
        return "\n".join(report)
    
    def _interpret_effect_size(self, cohens_d: float) -> str:
        """Interpret Cohen's d effect size"""
        abs_d = abs(cohens_d)
        if abs_d < 0.2:
            return "NEGLIGIBLE"
        elif abs_d < 0.5:
            return "SMALL"
        elif abs_d < 0.8:
            return "MEDIUM"
        else:
            return "LARGE"
    
    def plot_impact_visualization(self, impacts: List[ImpactMetrics], 
                                save_path: Optional[str] = None) -> None:
        """Create comprehensive impact visualization"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Quantitative Change Point Impact Analysis', fontsize=16, fontweight='bold')
        
        # Plot 1: Time series with change points and confidence bands
        ax1 = axes[0, 0]
        dates = pd.to_datetime(self.data['date'])
        ax1.plot(dates, self.prices, 'b-', alpha=0.7, linewidth=1, label='Oil Prices')
        
        for impact in impacts:
            cp_date = pd.to_datetime(impact.change_point_date)
            ax1.axvline(cp_date, color='red', linestyle='--', alpha=0.8, linewidth=2)
            
            # Add confidence bands around change point
            cp_idx = impact.change_point_index
            if cp_idx > 30 and cp_idx < len(self.prices) - 30:
                before_mean = impact.mean_before
                after_mean = impact.mean_after
                
                # Shade before/after regions
                before_dates = dates[:cp_idx]
                after_dates = dates[cp_idx:]
                
                ax1.fill_between(before_dates, before_mean - impact.volatility_before * 50,
                               before_mean + impact.volatility_before * 50, 
                               alpha=0.2, color='blue', label='Before ±1σ')
                ax1.fill_between(after_dates, after_mean - impact.volatility_after * 50,
                               after_mean + impact.volatility_after * 50,
                               alpha=0.2, color='orange', label='After ±1σ')
        
        ax1.set_title('Price Series with Change Points & Uncertainty Bands')
        ax1.set_ylabel('Price ($)')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Mean change impacts
        ax2 = axes[0, 1]
        change_dates = [imp.change_point_date for imp in impacts]
        mean_changes = [imp.mean_change_percent for imp in impacts]
        colors = ['green' if x > 0 else 'red' for x in mean_changes]
        
        bars = ax2.bar(range(len(impacts)), mean_changes, color=colors, alpha=0.7)
        ax2.set_title('Mean Price Change at Each Change Point')
        ax2.set_ylabel('Change (%)')
        ax2.set_xlabel('Change Point')
        ax2.set_xticks(range(len(impacts)))
        ax2.set_xticklabels([f"CP{i+1}" for i in range(len(impacts))])
        ax2.grid(True, alpha=0.3)
        ax2.axhline(y=0, color='black', linestyle='-', alpha=0.5)
        
        # Add value labels on bars
        for bar, value in zip(bars, mean_changes):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height + (1 if height > 0 else -3),
                    f'{value:.1f}%', ha='center', va='bottom' if height > 0 else 'top')
        
        # Plot 3: Volatility changes
        ax3 = axes[1, 0]
        vol_changes = [imp.volatility_change_percent for imp in impacts]
        colors = ['orange' if x > 0 else 'blue' for x in vol_changes]
        
        bars = ax3.bar(range(len(impacts)), vol_changes, color=colors, alpha=0.7)
        ax3.set_title('Volatility Change at Each Change Point')
        ax3.set_ylabel('Volatility Change (%)')
        ax3.set_xlabel('Change Point')
        ax3.set_xticks(range(len(impacts)))
        ax3.set_xticklabels([f"CP{i+1}" for i in range(len(impacts))])
        ax3.grid(True, alpha=0.3)
        ax3.axhline(y=0, color='black', linestyle='-', alpha=0.5)
        
        # Plot 4: Statistical significance
        ax4 = axes[1, 1]
        p_values = [imp.p_value for imp in impacts]
        effect_sizes = [abs(imp.effect_size_cohens_d) for imp in impacts]
        
        scatter = ax4.scatter(p_values, effect_sizes, 
                            c=['red' if p < 0.05 else 'gray' for p in p_values],
                            s=100, alpha=0.7)
        ax4.axvline(x=0.05, color='red', linestyle='--', alpha=0.5, label='p = 0.05')
        ax4.axhline(y=0.5, color='orange', linestyle='--', alpha=0.5, label='Medium Effect')
        ax4.set_xlabel('P-value')
        ax4.set_ylabel('Effect Size (|Cohen\'s d|)')
        ax4.set_title('Statistical Significance vs Effect Size')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        # Add annotations
        for i, (p, es) in enumerate(zip(p_values, effect_sizes)):
            ax4.annotate(f'CP{i+1}', (p, es), xytext=(5, 5), 
                        textcoords='offset points', fontsize=8)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def export_impact_metrics(self, impacts: List[ImpactMetrics], 
                            filepath: str) -> None:
        """Export impact metrics to CSV"""
        data = []
        for impact in impacts:
            data.append({
                'change_point_date': impact.change_point_date,
                'change_point_index': impact.change_point_index,
                'mean_before': impact.mean_before,
                'mean_after': impact.mean_after,
                'mean_change_absolute': impact.mean_change_absolute,
                'mean_change_percent': impact.mean_change_percent,
                'variance_before': impact.variance_before,
                'variance_after': impact.variance_after,
                'variance_change_percent': impact.variance_change_percent,
                'volatility_before': impact.volatility_before,
                'volatility_after': impact.volatility_after,
                'volatility_change_percent': impact.volatility_change_percent,
                't_statistic': impact.t_statistic,
                'p_value': impact.p_value,
                'effect_size_cohens_d': impact.effect_size_cohens_d,
                'mean_diff_ci_lower': impact.mean_diff_ci_lower,
                'mean_diff_ci_upper': impact.mean_diff_ci_upper,
                'statistical_significance': 'Significant' if impact.p_value < 0.05 else 'Not Significant'
            })
        
        df = pd.DataFrame(data)
        df.to_csv(filepath, index=False)
        print(f"Impact metrics exported to: {filepath}")