/**
 * PriceChart Component
 * Interactive chart showing oil prices with events and change points overlay
 * Integrates data from Tasks 1 & 2 analysis
 */

import React from 'react';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer, ReferenceLine, Scatter } from 'recharts';

const PriceChart = ({ data, events, changePoints }) => {
  /**
   * Custom tooltip to show price and nearby events
   */
  const CustomTooltip = ({ active, payload, label }) => {
    if (active && payload && payload.length) {
      const price = payload[0].value;
      const date = label;
      
      // Find events near this date (within 7 days)
      const nearbyEvents = events.filter(event => {
        const eventDate = new Date(event.date);
        const chartDate = new Date(date);
        const diffDays = Math.abs((eventDate - chartDate) / (1000 * 60 * 60 * 24));
        return diffDays <= 7;
      });

      return (
        <div className="custom-tooltip">
          <p className="tooltip-date">{`Date: ${date}`}</p>
          <p className="tooltip-price">{`Price: $${price.toFixed(2)}`}</p>
          {nearbyEvents.length > 0 && (
            <div className="tooltip-events">
              <p><strong>Nearby Events:</strong></p>
              {nearbyEvents.map((event, idx) => (
                <p key={idx} className={`event-${event.impact.toLowerCase()}`}>
                  • {event.event} ({event.date})
                </p>
              ))}
            </div>
          )}
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
      
      return data.map(item => ({
        date: item.date,
        price: parseFloat(item.price),
        year: new Date(item.date).getFullYear()
      }));
    } catch (error) {
      console.error('Error formatting chart data:', error);
      return [];
    }
  };

  const chartData = formatData();

  if (chartData.length === 0) {
    return (
      <div className="chart-error">
        <p>No data available for chart display</p>
      </div>
    );
  }

  return (
    <div className="price-chart-container">
      <ResponsiveContainer width="100%" height={400}>
        <LineChart data={chartData} margin={{ top: 5, right: 30, left: 20, bottom: 5 }}>
          <CartesianGrid strokeDasharray="3 3" opacity={0.3} />
          <XAxis 
            dataKey="date" 
            tick={{ fontSize: 12 }}
            tickFormatter={(value) => {
              try {
                return new Date(value).getFullYear().toString();
              } catch {
                return value;
              }
            }}
          />
          <YAxis 
            tick={{ fontSize: 12 }}
            tickFormatter={(value) => `$${value}`}
          />
          <Tooltip content={<CustomTooltip />} />
          <Legend />
          
          {/* Main price line */}
          <Line 
            type="monotone" 
            dataKey="price" 
            stroke="#1e3c72" 
            strokeWidth={2}
            dot={false}
            name="Brent Oil Price"
          />
          
          {/* Change points as vertical lines */}
          {changePoints && changePoints.map((cp, index) => (
            <ReferenceLine 
              key={`cp-${index}`}
              x={cp.date} 
              stroke="#dc3545" 
              strokeDasharray="5 5"
              strokeWidth={2}
              label={{ 
                value: `Change Point ${index + 1}`, 
                position: 'topLeft',
                style: { fontSize: '10px', fill: '#dc3545' }
              }}
            />
          ))}
          
          {/* Major events as reference lines */}
          {events && events
            .filter(event => event.impact === 'High')
            .slice(0, 5) // Show only top 5 high-impact events to avoid clutter
            .map((event, index) => (
              <ReferenceLine 
                key={`event-${index}`}
                x={event.date} 
                stroke="#fd7e14" 
                strokeDasharray="2 2"
                strokeWidth={1}
                label={{ 
                  value: event.event.substring(0, 20) + '...', 
                  position: 'top',
                  style: { fontSize: '8px', fill: '#fd7e14' }
                }}
              />
            ))}
        </LineChart>
      </ResponsiveContainer>
      
      {/* Chart legend */}
      <div className="chart-legend">
        <div className="legend-item">
          <div className="legend-color" style={{ backgroundColor: '#1e3c72' }}></div>
          <span>Oil Price</span>
        </div>
        <div className="legend-item">
          <div className="legend-line" style={{ borderColor: '#dc3545', borderStyle: 'dashed' }}></div>
          <span>Change Points</span>
        </div>
        <div className="legend-item">
          <div className="legend-line" style={{ borderColor: '#fd7e14', borderStyle: 'dotted' }}></div>
          <span>Major Events</span>
        </div>
      </div>
    </div>
  );
};

export default PriceChart;