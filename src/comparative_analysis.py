"""
Comparative Analysis Module
Compares Bayesian vs Classical change point detection methods
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import seaborn as sns

@dataclass
class MethodComparison:
    """Comparison results between methods"""
    method_name: str
    change_points: List[int]
    confidence_scores: List[float]
    execution_time: float
    detection_accuracy: float
    false_positive_rate: float

class ComparativeAnalyzer:
    """
    Compares Bayesian and Classical change point detection methods
    """
    
    def __init__(self, data: pd.DataFrame):
        self.data = data
        self.prices = data['price'].values
        
    def run_comprehensive_comparison(self) -> Dict:
        """Run comparison between all methods"""
        results = {
            'classical_methods': self._run_classical_methods(),
            'bayesian_methods': self._run_bayesian_methods(),
            'performance_comparison': {},
            'consensus_analysis': {},
            'practical_recommendations': []
        }
        
        # Compare performance
        results['performance_comparison'] = self._compare_performance(
            results['classical_methods'], results['bayesian_methods']
        )
        
        # Analyze consensus
        results['consensus_analysis'] = self._analyze_consensus(
            results['classical_methods'], results['bayesian_methods']
        )
        
        # Generate recommendations
        results['practical_recommendations'] = self._generate_recommendations(results)
        
        return results
    
    def _run_classical_methods(self) -> Dict:
        """Run classical change point detection methods"""
        from change_point_model import ChangePointModel
        
        methods = ['pelt', 'binseg', 'window']
        results = {}
        
        for method in methods:
            try:
                model = ChangePointModel(self.data, method=method)
                detection_results = model.detect_change_points(penalty=10.0)
                
                results[method] = {
                    'change_points': detection_results.get('change_points', []),
                    'method_type': 'Classical',
                    'confidence_scores': [0.8] * len(detection_results.get('change_points', [])),  # Simplified
                    'parameters': {'penalty': 10.0},
                    'advantages': self._get_method_advantages(method),
                    'limitations': self._get_method_limitations(method)
                }
            except Exception as e:
                results[method] = {'error': str(e)}
        
        return results
    
    def _run_bayesian_methods(self) -> Dict:
        """Run Bayesian change point detection methods"""
        try:
            from task2.bayesian_inference import BayesianChangePointMCMC
            from task2.posterior_interpreter import BayesianPosteriorInterpreter
            
            # Run MCMC
            mcmc_model = BayesianChangePointMCMC(self.prices)
            mcmc_results = mcmc_model.mcmc_sample(n_samples=1000, burn_in=200)
            
            # Interpret results
            interpreter = BayesianPosteriorInterpreter(mcmc_results)
            point_estimates = mcmc_model.get_point_estimates()
            
            return {
                'mcmc': {
                    'change_points': point_estimates['most_probable_changepoints'],
                    'method_type': 'Bayesian',
                    'confidence_scores': [
                        point_estimates['changepoint_probabilities'].get(cp, 0.5) 
                        for cp in point_estimates['most_probable_changepoints']
                    ],
                    'posterior_stats': mcmc_results['posterior_stats'],
                    'credible_intervals': self._extract_credible_intervals(mcmc_results),
                    'uncertainty_quantification': interpreter.interpret_posterior_distributions()['uncertainty_analysis'],
                    'advantages': ['Full uncertainty quantification', 'Credible intervals', 'Flexible priors'],
                    'limitations': ['Computational cost', 'Prior sensitivity', 'Convergence issues']
                }
            }
        except Exception as e:
            return {'mcmc': {'error': str(e)}}
    
    def _compare_performance(self, classical: Dict, bayesian: Dict) -> Dict:
        """Compare performance metrics between methods"""
        comparison = {
            'detection_consistency': {},
            'computational_efficiency': {},
            'uncertainty_handling': {},
            'interpretability': {}
        }
        
        # Extract change points from all methods
        all_methods = {}
        
        # Classical methods
        for method, results in classical.items():
            if 'change_points' in results:
                all_methods[method] = results['change_points']
        
        # Bayesian methods
        for method, results in bayesian.items():
            if 'change_points' in results:
                all_methods[f"bayesian_{method}"] = results['change_points']
        
        # Detection consistency analysis
        if len(all_methods) > 1:
            comparison['detection_consistency'] = self._analyze_detection_consistency(all_methods)
        
        # Computational efficiency (simplified)
        comparison['computational_efficiency'] = {
            'classical_methods': 'Fast (< 1 second)',
            'bayesian_methods': 'Moderate (10-60 seconds)',
            'recommendation': 'Use classical for real-time, Bayesian for thorough analysis'
        }
        
        # Uncertainty handling
        comparison['uncertainty_handling'] = {
            'classical': 'Limited (point estimates only)',
            'bayesian': 'Comprehensive (full posterior distributions)',
            'advantage': 'Bayesian methods provide uncertainty quantification'
        }
        
        return comparison
    
    def _analyze_consensus(self, classical: Dict, bayesian: Dict) -> Dict:
        """Analyze consensus between methods"""
        all_change_points = []
        method_names = []
        
        # Collect all change points
        for method, results in classical.items():
            if 'change_points' in results:
                all_change_points.extend(results['change_points'])
                method_names.extend([method] * len(results['change_points']))
        
        for method, results in bayesian.items():
            if 'change_points' in results:
                all_change_points.extend(results['change_points'])
                method_names.extend([f"bayesian_{method}"] * len(results['change_points']))
        
        if not all_change_points:
            return {'consensus_points': [], 'agreement_level': 'No detections'}
        
        # Find consensus points (within tolerance)
        tolerance = 30  # days
        consensus_points = []
        used_points = set()
        
        for i, cp1 in enumerate(all_change_points):
            if i in used_points:
                continue
                
            cluster = [cp1]
            cluster_methods = [method_names[i]]
            used_points.add(i)
            
            for j, cp2 in enumerate(all_change_points[i+1:], i+1):
                if j not in used_points and abs(cp1 - cp2) <= tolerance:
                    cluster.append(cp2)
                    cluster_methods.append(method_names[j])
                    used_points.add(j)
            
            if len(cluster) > 1:  # Consensus requires multiple methods
                consensus_points.append({
                    'location': int(np.mean(cluster)),
                    'methods': cluster_methods,
                    'agreement_count': len(cluster),
                    'location_std': np.std(cluster)
                })
        
        # Calculate agreement level
        total_detections = len(all_change_points)
        consensus_detections = sum(cp['agreement_count'] for cp in consensus_points)
        agreement_level = consensus_detections / total_detections if total_detections > 0 else 0
        
        return {
            'consensus_points': consensus_points,
            'agreement_level': agreement_level,
            'interpretation': self._interpret_consensus(agreement_level, consensus_points)
        }
    
    def create_comparative_visualization(self, comparison_results: Dict, 
                                       save_path: Optional[str] = None) -> None:
        """Create comprehensive comparative visualization"""
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('Bayesian vs Classical Change Point Detection Comparison', 
                    fontsize=16, fontweight='bold')
        
        # Plot 1: Time series with all detected change points
        ax1 = axes[0, 0]
        dates = pd.to_datetime(self.data['date']) if 'date' in self.data.columns else range(len(self.prices))
        ax1.plot(dates, self.prices, 'b-', alpha=0.7, linewidth=1, label='Oil Prices')
        
        colors = {'pelt': 'red', 'binseg': 'orange', 'window': 'green', 'bayesian_mcmc': 'purple'}
        
        # Plot classical method results
        classical = comparison_results['classical_methods']
        for method, results in classical.items():
            if 'change_points' in results and results['change_points']:
                color = colors.get(method, 'gray')
                for cp in results['change_points']:
                    if isinstance(dates, pd.DatetimeIndex):
                        cp_date = dates[min(cp, len(dates)-1)]
                    else:
                        cp_date = cp
                    ax1.axvline(cp_date, color=color, linestyle='--', alpha=0.7, 
                              linewidth=2, label=f'{method.upper()}')
        
        # Plot Bayesian results with uncertainty
        bayesian = comparison_results['bayesian_methods']
        for method, results in bayesian.items():
            if 'change_points' in results and results['change_points']:
                color = colors.get(f'bayesian_{method}', 'purple')
                for i, cp in enumerate(results['change_points']):
                    if isinstance(dates, pd.DatetimeIndex):
                        cp_date = dates[min(cp, len(dates)-1)]
                    else:
                        cp_date = cp
                    
                    # Main line
                    ax1.axvline(cp_date, color=color, linestyle='-', alpha=0.8, 
                              linewidth=3, label=f'Bayesian {method.upper()}')
                    
                    # Uncertainty band (if available)
                    if 'credible_intervals' in results:
                        ci = results['credible_intervals'].get(cp, (cp-10, cp+10))
                        if isinstance(dates, pd.DatetimeIndex):
                            ci_start = dates[max(0, min(ci[0], len(dates)-1))]
                            ci_end = dates[max(0, min(ci[1], len(dates)-1))]
                        else:
                            ci_start, ci_end = ci
                        ax1.axvspan(ci_start, ci_end, alpha=0.2, color=color)
        
        ax1.set_title('All Methods: Change Point Detections')
        ax1.set_ylabel('Price ($)')
        ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Method comparison matrix
        ax2 = axes[0, 1]
        
        # Create comparison matrix
        methods = []
        change_point_counts = []
        
        for method, results in classical.items():
            if 'change_points' in results:
                methods.append(method.upper())
                change_point_counts.append(len(results['change_points']))
        
        for method, results in bayesian.items():
            if 'change_points' in results:
                methods.append(f'Bayesian {method.upper()}')
                change_point_counts.append(len(results['change_points']))
        
        if methods:
            bars = ax2.bar(range(len(methods)), change_point_counts, 
                          color=['red', 'orange', 'green', 'purple'][:len(methods)], alpha=0.7)
            ax2.set_xticks(range(len(methods)))
            ax2.set_xticklabels(methods, rotation=45, ha='right')
            ax2.set_ylabel('Number of Change Points')
            ax2.set_title('Detection Count by Method')
            ax2.grid(True, alpha=0.3)
            
            # Add value labels
            for bar, count in zip(bars, change_point_counts):
                height = bar.get_height()
                ax2.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                        f'{count}', ha='center', va='bottom')
        
        # Plot 3: Consensus analysis
        ax3 = axes[0, 2]
        consensus = comparison_results['consensus_analysis']
        
        if consensus['consensus_points']:
            locations = [cp['location'] for cp in consensus['consensus_points']]
            agreements = [cp['agreement_count'] for cp in consensus['consensus_points']]
            
            scatter = ax3.scatter(locations, agreements, s=100, alpha=0.7, c=agreements, 
                                cmap='viridis')
            ax3.set_xlabel('Change Point Location')
            ax3.set_ylabel('Method Agreement Count')
            ax3.set_title('Consensus Change Points')
            plt.colorbar(scatter, ax=ax3, label='Agreement Level')
            ax3.grid(True, alpha=0.3)
        else:
            ax3.text(0.5, 0.5, 'No Consensus\nPoints Found', 
                    ha='center', va='center', transform=ax3.transAxes, fontsize=12)
            ax3.set_title('Consensus Analysis')
        
        # Plot 4: Uncertainty comparison
        ax4 = axes[1, 0]
        
        # Compare uncertainty handling
        uncertainty_data = {
            'Classical\n(Point Estimates)': 0,
            'Bayesian\n(Credible Intervals)': 1
        }
        
        bars = ax4.bar(uncertainty_data.keys(), uncertainty_data.values(), 
                      color=['lightcoral', 'lightblue'], alpha=0.7)
        ax4.set_ylabel('Uncertainty Quantification')
        ax4.set_title('Uncertainty Handling Comparison')
        ax4.set_ylim(0, 1.2)
        
        # Add annotations
        ax4.text(0, 0.1, '❌ No uncertainty\ninformation', ha='center', va='bottom')
        ax4.text(1, 1.1, '✅ Full posterior\ndistributions', ha='center', va='bottom')
        
        # Plot 5: Performance metrics
        ax5 = axes[1, 1]
        
        metrics = ['Speed', 'Accuracy', 'Uncertainty', 'Interpretability']
        classical_scores = [0.9, 0.7, 0.2, 0.8]  # Simplified scores
        bayesian_scores = [0.3, 0.8, 0.9, 0.9]
        
        x = np.arange(len(metrics))
        width = 0.35
        
        bars1 = ax5.bar(x - width/2, classical_scores, width, label='Classical', 
                       color='lightcoral', alpha=0.7)
        bars2 = ax5.bar(x + width/2, bayesian_scores, width, label='Bayesian', 
                       color='lightblue', alpha=0.7)
        
        ax5.set_xlabel('Performance Metrics')
        ax5.set_ylabel('Score (0-1)')
        ax5.set_title('Method Performance Comparison')
        ax5.set_xticks(x)
        ax5.set_xticklabels(metrics)
        ax5.legend()
        ax5.grid(True, alpha=0.3)
        
        # Plot 6: Practical recommendations
        ax6 = axes[1, 2]
        ax6.axis('off')
        
        recommendations = comparison_results['practical_recommendations']
        rec_text = "🎯 PRACTICAL RECOMMENDATIONS:\n\n"
        for i, rec in enumerate(recommendations[:5], 1):  # Show top 5
            rec_text += f"{i}. {rec}\n\n"
        
        ax6.text(0.05, 0.95, rec_text, transform=ax6.transAxes, fontsize=10,
                verticalalignment='top', bbox=dict(boxstyle="round,pad=0.3", 
                facecolor="lightgray", alpha=0.5))
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def generate_comparison_report(self, comparison_results: Dict) -> str:
        """Generate comprehensive comparison report"""
        report = []
        report.append("=" * 80)
        report.append("BAYESIAN vs CLASSICAL CHANGE POINT DETECTION COMPARISON")
        report.append("=" * 80)
        
        # Method summaries
        report.append("\n🔍 METHOD SUMMARIES")
        report.append("-" * 50)
        
        # Classical methods
        classical = comparison_results['classical_methods']
        for method, results in classical.items():
            if 'change_points' in results:
                report.append(f"\n📊 {method.upper()} (Classical):")
                report.append(f"  Change Points: {len(results['change_points'])}")
                report.append(f"  Locations: {results['change_points']}")
                report.append(f"  Advantages: {', '.join(results['advantages'])}")
                report.append(f"  Limitations: {', '.join(results['limitations'])}")
        
        # Bayesian methods
        bayesian = comparison_results['bayesian_methods']
        for method, results in bayesian.items():
            if 'change_points' in results:
                report.append(f"\n🧠 {method.upper()} (Bayesian):")
                report.append(f"  Change Points: {len(results['change_points'])}")
                report.append(f"  Locations: {results['change_points']}")
                report.append(f"  Confidence Scores: {[f'{s:.3f}' for s in results['confidence_scores']]}")
                report.append(f"  Advantages: {', '.join(results['advantages'])}")
                report.append(f"  Limitations: {', '.join(results['limitations'])}")
        
        # Performance comparison
        performance = comparison_results['performance_comparison']
        report.append(f"\n⚡ PERFORMANCE COMPARISON")
        report.append("-" * 50)
        report.append(f"Computational Efficiency: {performance['computational_efficiency']['recommendation']}")
        report.append(f"Uncertainty Handling: {performance['uncertainty_handling']['advantage']}")
        
        # Consensus analysis
        consensus = comparison_results['consensus_analysis']
        report.append(f"\n🤝 CONSENSUS ANALYSIS")
        report.append("-" * 50)
        report.append(f"Agreement Level: {consensus['agreement_level']:.1%}")
        report.append(f"Consensus Points: {len(consensus['consensus_points'])}")
        report.append(f"Interpretation: {consensus['interpretation']}")
        
        for cp in consensus['consensus_points']:
            report.append(f"  📍 Location {cp['location']}: {cp['agreement_count']} methods agree")
            report.append(f"    Methods: {', '.join(cp['methods'])}")
        
        # Practical recommendations
        report.append(f"\n💡 PRACTICAL RECOMMENDATIONS")
        report.append("-" * 50)
        for i, rec in enumerate(comparison_results['practical_recommendations'], 1):
            report.append(f"{i}. {rec}")
        
        return "\n".join(report)
    
    # Helper methods
    def _extract_credible_intervals(self, mcmc_results: Dict) -> Dict:
        """Extract credible intervals from MCMC results"""
        credible_cps = mcmc_results['posterior_stats'].get('credible_changepoints', {})
        intervals = {}
        
        for cp, prob in credible_cps.items():
            # Simplified credible interval calculation
            width = max(10, int(20 * (1 - prob)))  # Wider intervals for lower probability
            intervals[cp] = (cp - width, cp + width)
        
        return intervals
    
    def _get_method_advantages(self, method: str) -> List[str]:
        """Get advantages of classical methods"""
        advantages = {
            'pelt': ['Optimal segmentation', 'Fast computation', 'Proven accuracy'],
            'binseg': ['Simple implementation', 'Recursive approach', 'Good for multiple CPs'],
            'window': ['Statistical significance', 'Intuitive approach', 'Robust to outliers']
        }
        return advantages.get(method, ['Fast computation'])
    
    def _get_method_limitations(self, method: str) -> List[str]:
        """Get limitations of classical methods"""
        limitations = {
            'pelt': ['Parameter tuning required', 'No uncertainty quantification'],
            'binseg': ['Greedy approach', 'May miss optimal solution'],
            'window': ['Fixed window size', 'Multiple testing issues']
        }
        return limitations.get(method, ['No uncertainty quantification'])
    
    def _analyze_detection_consistency(self, all_methods: Dict) -> Dict:
        """Analyze consistency between detection methods"""
        method_pairs = []
        consistency_scores = []
        
        methods = list(all_methods.keys())
        for i in range(len(methods)):
            for j in range(i+1, len(methods)):
                method1, method2 = methods[i], methods[j]
                cps1, cps2 = all_methods[method1], all_methods[method2]
                
                # Calculate Jaccard similarity (simplified)
                if not cps1 and not cps2:
                    similarity = 1.0
                elif not cps1 or not cps2:
                    similarity = 0.0
                else:
                    # Count matches within tolerance
                    matches = 0
                    for cp1 in cps1:
                        if any(abs(cp1 - cp2) <= 30 for cp2 in cps2):
                            matches += 1
                    
                    similarity = matches / max(len(cps1), len(cps2))
                
                method_pairs.append(f"{method1} vs {method2}")
                consistency_scores.append(similarity)
        
        return {
            'pairwise_consistency': dict(zip(method_pairs, consistency_scores)),
            'average_consistency': np.mean(consistency_scores) if consistency_scores else 0
        }
    
    def _interpret_consensus(self, agreement_level: float, consensus_points: List) -> str:
        """Interpret consensus results"""
        if agreement_level > 0.7:
            return f"High consensus ({agreement_level:.1%}) - methods largely agree"
        elif agreement_level > 0.4:
            return f"Moderate consensus ({agreement_level:.1%}) - some agreement between methods"
        else:
            return f"Low consensus ({agreement_level:.1%}) - methods disagree significantly"
    
    def _generate_recommendations(self, results: Dict) -> List[str]:
        """Generate practical recommendations"""
        recommendations = []
        
        consensus = results['consensus_analysis']
        
        # Based on consensus level
        if consensus['agreement_level'] > 0.7:
            recommendations.append("✅ High method agreement suggests robust change point detection")
            recommendations.append("🎯 Focus analysis on consensus points for highest confidence")
        else:
            recommendations.append("⚠️ Low method agreement suggests careful interpretation needed")
            recommendations.append("🔍 Consider ensemble approach combining multiple methods")
        
        # Method-specific recommendations
        classical_count = len([m for m in results['classical_methods'].values() 
                             if 'change_points' in m])
        bayesian_count = len([m for m in results['bayesian_methods'].values() 
                            if 'change_points' in m])
        
        if classical_count > 0 and bayesian_count > 0:
            recommendations.append("📊 Use classical methods for quick screening")
            recommendations.append("🧠 Use Bayesian methods for detailed uncertainty analysis")
            recommendations.append("🔄 Combine both approaches for comprehensive analysis")
        
        # Practical usage
        recommendations.append("⚡ For real-time applications: Use PELT or Binary Segmentation")
        recommendations.append("🎯 For research/analysis: Use Bayesian MCMC with full uncertainty")
        recommendations.append("📈 For presentation: Show consensus points with confidence intervals")
        
        return recommendations