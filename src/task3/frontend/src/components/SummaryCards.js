/**
 * SummaryCards Component
 * Displays key metrics and statistics from all tasks analysis
 * Provides overview of data coverage, analysis results, and insights
 */

import React from 'react';

const SummaryCards = ({ summary }) => {
  /**
   * Format large numbers for display
   */
  const formatNumber = (num) => {
    try {
      if (num >= 1000000) {
        return (num / 1000000).toFixed(1) + 'M';
      } else if (num >= 1000) {
        return (num / 1000).toFixed(1) + 'K';
      }
      return num?.toLocaleString() || '0';
    } catch (error) {
      console.error('Error formatting number:', error);
      return '0';
    }
  };

  /**
   * Format currency values
   */
  const formatCurrency = (value) => {
    try {
      return `$${parseFloat(value).toFixed(2)}`;
    } catch (error) {
      return '$0.00';
    }
  };

  /**
   * Format percentage values
   */
  const formatPercentage = (value) => {
    try {
      return `${parseFloat(value).toFixed(1)}%`;
    } catch (error) {
      return '0.0%';
    }
  };

  /**
   * Get trend indicator based on value
   */
  const getTrendIndicator = (current, previous) => {
    try {
      if (!current || !previous) return null;
      
      const change = ((current - previous) / previous) * 100;
      if (Math.abs(change) < 0.1) return { icon: '→', class: 'neutral', text: 'No change' };
      if (change > 0) return { icon: '↗', class: 'positive', text: `+${change.toFixed(1)}%` };
      return { icon: '↘', class: 'negative', text: `${change.toFixed(1)}%` };
    } catch (error) {
      return null;
    }
  };

  // Default summary structure if not provided
  const defaultSummary = {
    data_overview: {
      total_observations: 0,
      date_range: { start: 'N/A', end: 'N/A' },
      years_covered: 0
    },
    price_statistics: {
      min: 0, max: 0, mean: 0, median: 0, std: 0, volatility: 0
    },
    analysis_results: {
      change_points_detected: 0,
      events_analyzed: 0,
      high_impact_events: 0
    },
    event_categories: {}
  };

  const data = summary || defaultSummary;

  /**
   * Define summary cards configuration
   */
  const cards = [
    {
      id: 'data-coverage',
      title: 'Data Coverage',
      icon: '📊',
      metrics: [
        {
          label: 'Total Observations',
          value: formatNumber(data.data_overview?.total_observations),
          subtitle: `${data.data_overview?.years_covered || 0} years of data`
        },
        {
          label: 'Date Range',
          value: data.data_overview?.date_range?.start ? 
            `${new Date(data.data_overview.date_range.start).getFullYear()} - ${new Date(data.data_overview.date_range.end).getFullYear()}` : 
            'N/A',
          subtitle: 'Historical coverage'
        }
      ]
    },
    {
      id: 'price-stats',
      title: 'Price Statistics',
      icon: '💰',
      metrics: [
        {
          label: 'Current Range',
          value: `${formatCurrency(data.price_statistics?.min)} - ${formatCurrency(data.price_statistics?.max)}`,
          subtitle: 'Min - Max prices'
        },
        {
          label: 'Average Price',
          value: formatCurrency(data.price_statistics?.mean),
          subtitle: `Median: ${formatCurrency(data.price_statistics?.median)}`
        },
        {
          label: 'Volatility',
          value: formatPercentage((data.price_statistics?.volatility || 0) * 100),
          subtitle: `Std Dev: ${formatCurrency(data.price_statistics?.std)}`
        }
      ]
    },
    {
      id: 'analysis-results',
      title: 'Analysis Results',
      icon: '🔍',
      metrics: [
        {
          label: 'Change Points',
          value: data.analysis_results?.change_points_detected || 0,
          subtitle: 'Structural breaks detected'
        },
        {
          label: 'Events Analyzed',
          value: data.analysis_results?.events_analyzed || 0,
          subtitle: `${data.analysis_results?.high_impact_events || 0} high impact`
        }
      ]
    },
    {
      id: 'event-breakdown',
      title: 'Event Categories',
      icon: '🌍',
      metrics: Object.entries(data.event_categories || {}).map(([category, count]) => ({
        label: category,
        value: count,
        subtitle: `${((count / (data.analysis_results?.events_analyzed || 1)) * 100).toFixed(0)}% of total`
      }))
    }
  ];

  return (
    <div className="summary-cards">
      {cards.map(card => (
        <div key={card.id} className={`summary-card ${card.id}`}>
          <div className="card-header">
            <span className="card-icon" role="img" aria-label={card.title}>
              {card.icon}
            </span>
            <h4 className="card-title">{card.title}</h4>
          </div>
          
          <div className="card-content">
            {card.metrics.map((metric, index) => (
              <div key={index} className="metric">
                <div className="metric-value">
                  {metric.value}
                </div>
                <div className="metric-label">
                  {metric.label}
                </div>
                {metric.subtitle && (
                  <div className="metric-subtitle">
                    {metric.subtitle}
                  </div>
                )}
              </div>
            ))}
          </div>
        </div>
      ))}

      {/* Additional insights card */}
      <div className="summary-card insights-card">
        <div className="card-header">
          <span className="card-icon" role="img" aria-label="Key Insights">💡</span>
          <h4 className="card-title">Key Insights</h4>
        </div>
        
        <div className="card-content">
          <div className="insights-list">
            <div className="insight-item">
              <span className="insight-icon">📈</span>
              <span className="insight-text">
                {data.analysis_results?.change_points_detected > 0 ? 
                  `${data.analysis_results.change_points_detected} major structural breaks identified` :
                  'No significant structural breaks detected'
                }
              </span>
            </div>
            
            <div className="insight-item">
              <span className="insight-icon">⚡</span>
              <span className="insight-text">
                {data.price_statistics?.volatility > 0.02 ? 
                  'High volatility period detected' :
                  'Relatively stable price environment'
                }
              </span>
            </div>
            
            <div className="insight-item">
              <span className="insight-icon">🎯</span>
              <span className="insight-text">
                {data.analysis_results?.high_impact_events > 0 ?
                  `${data.analysis_results.high_impact_events} high-impact events may correlate with price changes` :
                  'Limited high-impact events in analysis period'
                }
              </span>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default SummaryCards;