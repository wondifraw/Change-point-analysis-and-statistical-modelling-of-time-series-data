/**
 * ChangePointsChart Component
 * Visualizes change points from Task 2 Bayesian analysis
 * Shows confidence levels and impact metrics
 */

import React from 'react';
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer, Cell } from 'recharts';

const ChangePointsChart = ({ data }) => {
  /**
   * Custom tooltip for change points
   */
  const CustomTooltip = ({ active, payload, label }) => {
    if (active && payload && payload.length) {
      const changePoint = payload[0].payload;
      
      return (
        <div className="changepoint-tooltip">
          <p className="tooltip-title"><strong>Change Point Analysis</strong></p>
          <p><strong>Date:</strong> {label}</p>
          <p><strong>Confidence:</strong> {(changePoint.confidence * 100).toFixed(1)}%</p>
          <p><strong>Price Before:</strong> ${changePoint.before_mean?.toFixed(2) || 'N/A'}</p>
          <p><strong>Price After:</strong> ${changePoint.after_mean?.toFixed(2) || 'N/A'}</p>
          <p><strong>Change:</strong> 
            <span className={changePoint.change_percent > 0 ? 'positive-change' : 'negative-change'}>
              {changePoint.change_percent > 0 ? '+' : ''}{changePoint.change_percent?.toFixed(1) || 'N/A'}%
            </span>
          </p>
          {changePoint.method && <p><strong>Method:</strong> {changePoint.method}</p>}
        </div>
      );
    }
    return null;
  };

  /**
   * Format data for chart display
   */
  const formatData = () => {
    try {
      if (!data || data.length === 0) return [];
      
      return data.map((cp, index) => ({
        ...cp,
        date: new Date(cp.date).toLocaleDateString('en-US', { 
          year: 'numeric', 
          month: 'short', 
          day: 'numeric' 
        }),
        confidencePercent: (cp.confidence || 0) * 100,
        absChangePercent: Math.abs(cp.change_percent || 0),
        changeDirection: (cp.change_percent || 0) > 0 ? 'increase' : 'decrease',
        index: index + 1
      }));
    } catch (error) {
      console.error('Error formatting change points data:', error);
      return [];
    }
  };

  /**
   * Get color based on change direction and magnitude
   */
  const getBarColor = (changePercent) => {
    if (changePercent > 0) {
      return changePercent > 20 ? '#dc3545' : '#fd7e14'; // Red for large increases, orange for moderate
    } else {
      return changePercent < -20 ? '#28a745' : '#17a2b8'; // Green for large decreases, blue for moderate
    }
  };

  const chartData = formatData();

  if (chartData.length === 0) {
    return (
      <div className="changepoints-chart-container">
        <div className="no-data-message">
          <p>No change points detected</p>
          <small>Run the Bayesian analysis to detect structural breaks</small>
        </div>
      </div>
    );
  }

  return (
    <div className="changepoints-chart-container">
      {/* Chart */}
      <ResponsiveContainer width="100%" height={300}>
        <BarChart data={chartData} margin={{ top: 20, right: 30, left: 20, bottom: 5 }}>
          <CartesianGrid strokeDasharray="3 3" opacity={0.3} />
          <XAxis 
            dataKey="date" 
            tick={{ fontSize: 10 }}
            angle={-45}
            textAnchor="end"
            height={80}
          />
          <YAxis 
            tick={{ fontSize: 12 }}
            label={{ value: 'Price Change (%)', angle: -90, position: 'insideLeft' }}
          />
          <Tooltip content={<CustomTooltip />} />
          <Legend />
          
          <Bar 
            dataKey="change_percent" 
            name="Price Change (%)"
            radius={[2, 2, 0, 0]}
          >
            {chartData.map((entry, index) => (
              <Cell key={`cell-${index}`} fill={getBarColor(entry.change_percent)} />
            ))}
          </Bar>
        </BarChart>
      </ResponsiveContainer>

      {/* Change points summary */}
      <div className="changepoints-summary">
        <h4>Change Points Summary</h4>
        <div className="summary-grid">
          {chartData.map((cp, index) => (
            <div key={index} className="changepoint-card">
              <div className="card-header">
                <span className="cp-number">#{cp.index}</span>
                <span className="cp-date">{cp.date}</span>
              </div>
              
              <div className="card-body">
                <div className="confidence-bar">
                  <div className="confidence-label">Confidence</div>
                  <div className="confidence-progress">
                    <div 
                      className="confidence-fill" 
                      style={{ 
                        width: `${cp.confidencePercent}%`,
                        backgroundColor: cp.confidencePercent > 90 ? '#28a745' : 
                                       cp.confidencePercent > 70 ? '#fd7e14' : '#dc3545'
                      }}
                    ></div>
                  </div>
                  <span className="confidence-value">{cp.confidencePercent.toFixed(1)}%</span>
                </div>
                
                <div className="price-change">
                  <span className="change-label">Price Impact:</span>
                  <span className={`change-value ${cp.changeDirection}`}>
                    {cp.change_percent > 0 ? '+' : ''}{cp.change_percent?.toFixed(1) || 'N/A'}%
                  </span>
                </div>
                
                {cp.method && (
                  <div className="method-info">
                    <span className="method-label">Method:</span>
                    <span className="method-value">{cp.method}</span>
                  </div>
                )}
              </div>
            </div>
          ))}
        </div>
      </div>

      {/* Analysis insights */}
      <div className="analysis-insights">
        <h4>Key Insights</h4>
        <ul>
          <li>
            <strong>{chartData.length}</strong> significant change points detected using Bayesian analysis
          </li>
          <li>
            Average confidence level: <strong>
              {(chartData.reduce((sum, cp) => sum + cp.confidencePercent, 0) / chartData.length).toFixed(1)}%
            </strong>
          </li>
          <li>
            Largest price impact: <strong>
              {Math.max(...chartData.map(cp => Math.abs(cp.change_percent))).toFixed(1)}%
            </strong>
          </li>
          <li>
            Most recent change point: <strong>
              {chartData.sort((a, b) => new Date(b.date) - new Date(a.date))[0]?.date || 'N/A'}
            </strong>
          </li>
        </ul>
      </div>
    </div>
  );
};

export default ChangePointsChart;