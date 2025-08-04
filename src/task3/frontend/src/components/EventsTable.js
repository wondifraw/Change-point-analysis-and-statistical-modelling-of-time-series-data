/**
 * EventsTable Component
 * Displays geopolitical events from Task 1 analysis in a sortable table
 * Includes filtering and impact visualization
 */

import React, { useState, useMemo } from 'react';

const EventsTable = ({ events }) => {
  const [sortConfig, setSortConfig] = useState({ key: 'date', direction: 'desc' });
  const [filterCategory, setFilterCategory] = useState('all');
  const [filterImpact, setFilterImpact] = useState('all');

  /**
   * Sort events based on current sort configuration
   */
  const sortedEvents = useMemo(() => {
    try {
      if (!events || events.length === 0) return [];

      let sortableEvents = [...events];
      
      // Apply filters
      if (filterCategory !== 'all') {
        sortableEvents = sortableEvents.filter(event => event.category === filterCategory);
      }
      if (filterImpact !== 'all') {
        sortableEvents = sortableEvents.filter(event => event.impact === filterImpact);
      }

      // Apply sorting
      if (sortConfig.key) {
        sortableEvents.sort((a, b) => {
          let aValue = a[sortConfig.key];
          let bValue = b[sortConfig.key];

          // Handle date sorting
          if (sortConfig.key === 'date') {
            aValue = new Date(aValue);
            bValue = new Date(bValue);
          }

          if (aValue < bValue) {
            return sortConfig.direction === 'asc' ? -1 : 1;
          }
          if (aValue > bValue) {
            return sortConfig.direction === 'asc' ? 1 : -1;
          }
          return 0;
        });
      }

      return sortableEvents;
    } catch (error) {
      console.error('Error sorting events:', error);
      return events || [];
    }
  }, [events, sortConfig, filterCategory, filterImpact]);

  /**
   * Handle column header click for sorting
   */
  const handleSort = (key) => {
    try {
      let direction = 'asc';
      if (sortConfig.key === key && sortConfig.direction === 'asc') {
        direction = 'desc';
      }
      setSortConfig({ key, direction });
    } catch (error) {
      console.error('Error handling sort:', error);
    }
  };

  /**
   * Get unique categories for filter dropdown
   */
  const getUniqueCategories = () => {
    try {
      if (!events || events.length === 0) return [];
      return [...new Set(events.map(event => event.category))];
    } catch (error) {
      console.error('Error getting categories:', error);
      return [];
    }
  };

  /**
   * Get unique impact levels for filter dropdown
   */
  const getUniqueImpacts = () => {
    try {
      if (!events || events.length === 0) return [];
      return [...new Set(events.map(event => event.impact))];
    } catch (error) {
      console.error('Error getting impacts:', error);
      return [];
    }
  };

  /**
   * Format date for display
   */
  const formatDate = (dateString) => {
    try {
      return new Date(dateString).toLocaleDateString('en-US', {
        year: 'numeric',
        month: 'short',
        day: 'numeric'
      });
    } catch (error) {
      return dateString;
    }
  };

  /**
   * Get CSS class for impact level
   */
  const getImpactClass = (impact) => {
    const impactClasses = {
      'High': 'impact-high',
      'Medium': 'impact-medium',
      'Low': 'impact-low'
    };
    return impactClasses[impact] || '';
  };

  /**
   * Get CSS class for category
   */
  const getCategoryClass = (category) => {
    const categoryClasses = {
      'Geopolitical': 'category-geopolitical',
      'Economic': 'category-economic',
      'OPEC Decision': 'category-opec'
    };
    return categoryClasses[category] || '';
  };

  if (!events || events.length === 0) {
    return (
      <div className="events-table-container">
        <p>No events data available</p>
      </div>
    );
  }

  return (
    <div className="events-table-container">
      {/* Filter controls */}
      <div className="table-filters">
        <div className="filter-group">
          <label htmlFor="category-filter">Category:</label>
          <select 
            id="category-filter"
            value={filterCategory} 
            onChange={(e) => setFilterCategory(e.target.value)}
          >
            <option value="all">All Categories</option>
            {getUniqueCategories().map(category => (
              <option key={category} value={category}>{category}</option>
            ))}
          </select>
        </div>

        <div className="filter-group">
          <label htmlFor="impact-filter">Impact:</label>
          <select 
            id="impact-filter"
            value={filterImpact} 
            onChange={(e) => setFilterImpact(e.target.value)}
          >
            <option value="all">All Impacts</option>
            {getUniqueImpacts().map(impact => (
              <option key={impact} value={impact}>{impact}</option>
            ))}
          </select>
        </div>

        <div className="results-count">
          Showing {sortedEvents.length} of {events.length} events
        </div>
      </div>

      {/* Events table */}
      <div className="table-wrapper">
        <table className="events-table">
          <thead>
            <tr>
              <th 
                onClick={() => handleSort('date')}
                className={`sortable ${sortConfig.key === 'date' ? sortConfig.direction : ''}`}
              >
                Date
                {sortConfig.key === 'date' && (
                  <span className="sort-indicator">
                    {sortConfig.direction === 'asc' ? ' ↑' : ' ↓'}
                  </span>
                )}
              </th>
              <th 
                onClick={() => handleSort('event')}
                className={`sortable ${sortConfig.key === 'event' ? sortConfig.direction : ''}`}
              >
                Event
                {sortConfig.key === 'event' && (
                  <span className="sort-indicator">
                    {sortConfig.direction === 'asc' ? ' ↑' : ' ↓'}
                  </span>
                )}
              </th>
              <th 
                onClick={() => handleSort('category')}
                className={`sortable ${sortConfig.key === 'category' ? sortConfig.direction : ''}`}
              >
                Category
                {sortConfig.key === 'category' && (
                  <span className="sort-indicator">
                    {sortConfig.direction === 'asc' ? ' ↑' : ' ↓'}
                  </span>
                )}
              </th>
              <th 
                onClick={() => handleSort('impact')}
                className={`sortable ${sortConfig.key === 'impact' ? sortConfig.direction : ''}`}
              >
                Impact
                {sortConfig.key === 'impact' && (
                  <span className="sort-indicator">
                    {sortConfig.direction === 'asc' ? ' ↑' : ' ↓'}
                  </span>
                )}
              </th>
            </tr>
          </thead>
          <tbody>
            {sortedEvents.map((event, index) => (
              <tr key={`${event.date}-${index}`} className="event-row">
                <td className="date-cell">
                  {formatDate(event.date)}
                </td>
                <td className="event-cell">
                  <div className="event-text" title={event.event}>
                    {event.event}
                  </div>
                </td>
                <td className="category-cell">
                  <span className={`category-badge ${getCategoryClass(event.category)}`}>
                    {event.category}
                  </span>
                </td>
                <td className="impact-cell">
                  <span className={`impact-badge ${getImpactClass(event.impact)}`}>
                    {event.impact}
                  </span>
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      {sortedEvents.length === 0 && (
        <div className="no-results">
          <p>No events match the current filters</p>
        </div>
      )}
    </div>
  );
};

export default EventsTable;