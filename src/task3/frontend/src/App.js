/**
 * Main Dashboard Application
 * Integrates all Tasks (1, 2, 3) analysis results into interactive dashboard
 * Provides comprehensive view of oil price analysis with error handling
 */

import React, { useState, useEffect, useCallback } from 'react';
import axios from 'axios';
import PriceChart from './components/PriceChart';
import EventsTable from './components/EventsTable';
import ChangePointsChart from './components/ChangePointsChart';
import FilterPanel from './components/FilterPanel';
import SummaryCards from './components/SummaryCards';
import './App.css';

// Configure axios defaults
axios.defaults.timeout = 10000; // 10 second timeout

function App() {
  // State management for all data and UI
  const [state, setState] = useState({
    oilPrices: [],
    events: [],
    changePoints: [],
    summary: {},
    eventImpactAnalysis: [],
    loading: true,
    error: null,
    lastUpdated: null
  });

  const [filters, setFilters] = useState({
    startDate: '',
    endDate: '',
    category: 'all',
    impact: 'all',
    priceRange: { min: '', max: '' }
  });

  const [availableFilters, setAvailableFilters] = useState({
    categories: [],
    impacts: []
  });

  /**
   * Load all data from backend APIs with comprehensive error handling
   */
  const loadData = useCallback(async () => {
    try {
      setState(prev => ({ ...prev, loading: true, error: null }));

      // Check backend health first
      const healthCheck = await axios.get('/api/health');
      console.log('Backend health:', healthCheck.data);

      // Load all data in parallel for better performance
      const [
        pricesRes,
        eventsRes,
        changePointsRes,
        summaryRes,
        impactAnalysisRes
      ] = await Promise.allSettled([
        axios.get('/api/oil-prices'),
        axios.get('/api/events'),
        axios.get('/api/change-points'),
        axios.get('/api/analysis-summary'),
        axios.get('/api/event-impact-analysis')
      ]);

      // Process results with error handling for each endpoint
      const newState = {
        loading: false,
        lastUpdated: new Date().toISOString(),
        error: null
      };

      // Process oil prices
      if (pricesRes.status === 'fulfilled' && pricesRes.value.data.success) {
        newState.oilPrices = pricesRes.value.data.data;
      } else {
        console.error('Failed to load oil prices:', pricesRes.reason);
        newState.oilPrices = [];
      }

      // Process events
      if (eventsRes.status === 'fulfilled' && eventsRes.value.data.success) {
        newState.events = eventsRes.value.data.data;
        // Update available filters
        setAvailableFilters(prev => ({
          ...prev,
          categories: eventsRes.value.data.filters?.categories || [],
          impacts: eventsRes.value.data.filters?.impacts || []
        }));
      } else {
        console.error('Failed to load events:', eventsRes.reason);
        newState.events = [];
      }

      // Process change points
      if (changePointsRes.status === 'fulfilled' && changePointsRes.value.data.success) {
        newState.changePoints = changePointsRes.value.data.data;
      } else {
        console.error('Failed to load change points:', changePointsRes.reason);
        newState.changePoints = [];
      }

      // Process summary
      if (summaryRes.status === 'fulfilled' && summaryRes.value.data.success) {
        newState.summary = summaryRes.value.data.data;
      } else {
        console.error('Failed to load summary:', summaryRes.reason);
        newState.summary = {};
      }

      // Process impact analysis
      if (impactAnalysisRes.status === 'fulfilled' && impactAnalysisRes.value.data.success) {
        newState.eventImpactAnalysis = impactAnalysisRes.value.data.data;
      } else {
        console.error('Failed to load impact analysis:', impactAnalysisRes.reason);
        newState.eventImpactAnalysis = [];
      }

      setState(prev => ({ ...prev, ...newState }));

    } catch (error) {
      console.error('Error loading data:', error);
      setState(prev => ({
        ...prev,
        loading: false,
        error: `Failed to load data: ${error.message}. Please check if the backend server is running.`
      }));
    }
  }, []);

  /**
   * Handle filter changes with API call
   */
  const handleFilterChange = useCallback(async (newFilters) => {
    try {
      setFilters(newFilters);

      // Build query parameters
      const params = new URLSearchParams();
      if (newFilters.startDate) params.append('start_date', newFilters.startDate);
      if (newFilters.endDate) params.append('end_date', newFilters.endDate);
      if (newFilters.category && newFilters.category !== 'all') {
        params.append('category', newFilters.category);
      }
      if (newFilters.impact && newFilters.impact !== 'all') {
        params.append('impact', newFilters.impact);
      }

      // Apply filters to oil prices
      const pricesResponse = await axios.get(`/api/oil-prices?${params}`);
      if (pricesResponse.data.success) {
        setState(prev => ({ ...prev, oilPrices: pricesResponse.data.data }));
      }

      // Apply filters to events
      const eventsResponse = await axios.get(`/api/events?${params}`);
      if (eventsResponse.data.success) {
        setState(prev => ({ ...prev, events: eventsResponse.data.data }));
      }

    } catch (error) {
      console.error('Error applying filters:', error);
      // Show user-friendly error message
      alert('Error applying filters. Please try again.');
    }
  }, []);

  /**
   * Refresh data manually
   */
  const refreshData = useCallback(() => {
    loadData();
  }, [loadData]);

  // Load data on component mount
  useEffect(() => {
    loadData();
  }, [loadData]);

  // Auto-refresh data every 5 minutes (optional)
  useEffect(() => {
    const interval = setInterval(() => {
      if (!state.loading) {
        console.log('Auto-refreshing data...');
        loadData();
      }
    }, 5 * 60 * 1000); // 5 minutes

    return () => clearInterval(interval);
  }, [loadData, state.loading]);

  /**
   * Render loading state
   */
  if (state.loading) {
    return (
      <div className="App">
        <div className="loading-container">
          <div className="loading-spinner"></div>
          <h2>Loading Oil Price Analysis Dashboard...</h2>
          <p>Fetching data from Tasks 1, 2, and 3 analysis...</p>
        </div>
      </div>
    );
  }

  /**
   * Render error state
   */
  if (state.error) {
    return (
      <div className="App">
        <div className="error-container">
          <h2>⚠️ Error Loading Dashboard</h2>
          <p>{state.error}</p>
          <div className="error-actions">
            <button onClick={refreshData} className="retry-btn">
              🔄 Retry Loading
            </button>
            <details className="error-details">
              <summary>Troubleshooting Tips</summary>
              <ul>
                <li>Ensure the Flask backend server is running on port 5000</li>
                <li>Check that all Task 1 and Task 2 modules are properly installed</li>
                <li>Verify the data files exist in the correct directories</li>
                <li>Check browser console for additional error details</li>
              </ul>
            </details>
          </div>
        </div>
      </div>
    );
  }

  return (
    <div className="App">
      {/* Header */}
      <header className="header">
        <div className="header-content">
          <h1>🛢️ Brent Oil Price Analysis Dashboard</h1>
          <p>Comprehensive analysis integrating change point detection, geopolitical events, and statistical modeling</p>
          <div className="header-meta">
            <span className="data-count">
              {state.oilPrices.length.toLocaleString()} price observations • {state.events.length} events • {state.changePoints.length} change points
            </span>
            {state.lastUpdated && (
              <span className="last-updated">
                Last updated: {new Date(state.lastUpdated).toLocaleTimeString()}
              </span>
            )}
            <button onClick={refreshData} className="refresh-btn" title="Refresh data">
              🔄
            </button>
          </div>
        </div>
      </header>

      {/* Main Dashboard */}
      <div className="dashboard">
        {/* Summary Cards */}
        <section className="dashboard-section">
          <SummaryCards summary={state.summary} />
        </section>
        
        {/* Filter Panel */}
        <section className="dashboard-section">
          <FilterPanel 
            filters={filters} 
            onFilterChange={handleFilterChange}
            availableCategories={availableFilters.categories}
            availableImpacts={availableFilters.impacts}
          />
        </section>

        {/* Charts Grid */}
        <section className="dashboard-section">
          <div className="charts-grid">
            <div className="chart-container main-chart">
              <div className="chart-header">
                <h3>📈 Oil Price Timeline with Events & Change Points</h3>
                <p>Interactive visualization showing price movements, geopolitical events, and detected structural breaks</p>
              </div>
              <PriceChart 
                data={state.oilPrices} 
                events={state.events} 
                changePoints={state.changePoints}
              />
            </div>

            <div className="chart-container">
              <div className="chart-header">
                <h3>🎯 Change Points Analysis</h3>
                <p>Bayesian detection results with confidence levels</p>
              </div>
              <ChangePointsChart data={state.changePoints} />
            </div>
          </div>
        </section>

        {/* Events Table */}
        <section className="dashboard-section">
          <div className="table-container">
            <div className="table-header">
              <h3>🌍 Geopolitical Events Analysis</h3>
              <p>Comprehensive list of events that may impact oil prices</p>
            </div>
            <EventsTable events={state.events} />
          </div>
        </section>

        {/* Event Impact Analysis */}
        {state.eventImpactAnalysis.length > 0 && (
          <section className="dashboard-section">
            <div className="impact-analysis-container">
              <div className="section-header">
                <h3>📊 Event Impact Analysis</h3>
                <p>Quantified price changes around major events</p>
              </div>
              
              <div className="impact-grid">
                {state.eventImpactAnalysis.slice(0, 6).map((analysis, index) => (
                  <div key={index} className="impact-card">
                    <div className="impact-header">
                      <h4>{analysis.event}</h4>
                      <span className={`impact-badge ${analysis.impact_level.toLowerCase()}`}>
                        {analysis.impact_level}
                      </span>
                    </div>
                    <div className="impact-metrics">
                      <div className="metric">
                        <span className="metric-label">Price Change:</span>
                        <span className={`metric-value ${analysis.change_percent > 0 ? 'positive' : 'negative'}`}>
                          {analysis.change_percent > 0 ? '+' : ''}{analysis.change_percent.toFixed(1)}%
                        </span>
                      </div>
                      <div className="metric">
                        <span className="metric-label">Date:</span>
                        <span className="metric-value">{analysis.date}</span>
                      </div>
                    </div>
                  </div>
                ))}
              </div>
            </div>
          </section>
        )}

        {/* Footer */}
        <footer className="dashboard-footer">
          <div className="footer-content">
            <p>
              <strong>Methodology:</strong> This dashboard integrates change point detection (Task 2), 
              geopolitical event analysis (Task 1), and interactive visualization (Task 3) to provide 
              comprehensive insights into Brent oil price dynamics.
            </p>
            <p>
              <strong>Disclaimer:</strong> Statistical correlations shown do not imply causation. 
              Analysis is for educational and research purposes only.
            </p>
          </div>
        </footer>
      </div>
    </div>
  );
}

export default App;