/**
 * FilterPanel Component
 * Provides interactive filters for date range, event categories, and analysis parameters
 * Integrates with all tasks' data filtering needs
 */

import React, { useState, useEffect } from 'react';

const FilterPanel = ({ filters, onFilterChange, availableCategories = [], availableImpacts = [] }) => {
  const [localFilters, setLocalFilters] = useState(filters);
  const [isExpanded, setIsExpanded] = useState(false);

  /**
   * Update local filters when props change
   */
  useEffect(() => {
    setLocalFilters(filters);
  }, [filters]);

  /**
   * Handle input changes with error handling
   */
  const handleInputChange = (field, value) => {
    try {
      setLocalFilters(prev => ({
        ...prev,
        [field]: value
      }));
    } catch (error) {
      console.error('Error updating filter:', error);
    }
  };

  /**
   * Apply filters with validation
   */
  const applyFilters = () => {
    try {
      // Validate date range
      if (localFilters.startDate && localFilters.endDate) {
        const startDate = new Date(localFilters.startDate);
        const endDate = new Date(localFilters.endDate);
        
        if (startDate > endDate) {
          alert('Start date must be before end date');
          return;
        }
      }

      onFilterChange(localFilters);
    } catch (error) {
      console.error('Error applying filters:', error);
      alert('Error applying filters. Please check your inputs.');
    }
  };

  /**
   * Reset filters to default values
   */
  const resetFilters = () => {
    try {
      const defaultFilters = {
        startDate: '',
        endDate: '',
        category: 'all',
        impact: 'all',
        priceRange: { min: '', max: '' }
      };
      
      setLocalFilters(defaultFilters);
      onFilterChange(defaultFilters);
    } catch (error) {
      console.error('Error resetting filters:', error);
    }
  };

  /**
   * Get preset date ranges for quick selection
   */
  const getPresetDateRanges = () => {
    const today = new Date();
    const ranges = {
      'Last Year': {
        start: new Date(today.getFullYear() - 1, today.getMonth(), today.getDate()),
        end: today
      },
      'Last 5 Years': {
        start: new Date(today.getFullYear() - 5, today.getMonth(), today.getDate()),
        end: today
      },
      '2008 Crisis': {
        start: new Date(2007, 0, 1),
        end: new Date(2010, 11, 31)
      },
      '2020 Pandemic': {
        start: new Date(2019, 0, 1),
        end: new Date(2021, 11, 31)
      },
      '2022 Ukraine War': {
        start: new Date(2021, 0, 1),
        end: new Date(2023, 11, 31)
      }
    };

    return ranges;
  };

  /**
   * Apply preset date range
   */
  const applyPresetRange = (rangeName) => {
    try {
      const ranges = getPresetDateRanges();
      const range = ranges[rangeName];
      
      if (range) {
        const updatedFilters = {
          ...localFilters,
          startDate: range.start.toISOString().split('T')[0],
          endDate: range.end.toISOString().split('T')[0]
        };
        
        setLocalFilters(updatedFilters);
        onFilterChange(updatedFilters);
      }
    } catch (error) {
      console.error('Error applying preset range:', error);
    }
  };

  return (
    <div className="filter-panel">
      <div className="filter-header">
        <h3>Data Filters</h3>
        <button 
          className="expand-toggle"
          onClick={() => setIsExpanded(!isExpanded)}
          aria-label={isExpanded ? 'Collapse filters' : 'Expand filters'}
        >
          {isExpanded ? '▼' : '▶'}
        </button>
      </div>

      <div className={`filter-content ${isExpanded ? 'expanded' : 'collapsed'}`}>
        {/* Date Range Filters */}
        <div className="filter-section">
          <h4>Date Range</h4>
          <div className="filter-row">
            <div className="filter-group">
              <label htmlFor="start-date">Start Date:</label>
              <input
                id="start-date"
                type="date"
                value={localFilters.startDate || ''}
                onChange={(e) => handleInputChange('startDate', e.target.value)}
                max={localFilters.endDate || undefined}
              />
            </div>

            <div className="filter-group">
              <label htmlFor="end-date">End Date:</label>
              <input
                id="end-date"
                type="date"
                value={localFilters.endDate || ''}
                onChange={(e) => handleInputChange('endDate', e.target.value)}
                min={localFilters.startDate || undefined}
              />
            </div>
          </div>

          {/* Preset date ranges */}
          <div className="preset-ranges">
            <label>Quick Select:</label>
            <div className="preset-buttons">
              {Object.keys(getPresetDateRanges()).map(rangeName => (
                <button
                  key={rangeName}
                  className="preset-btn"
                  onClick={() => applyPresetRange(rangeName)}
                  title={`Select ${rangeName} date range`}
                >
                  {rangeName}
                </button>
              ))}
            </div>
          </div>
        </div>

        {/* Event Filters */}
        <div className="filter-section">
          <h4>Event Filters</h4>
          <div className="filter-row">
            <div className="filter-group">
              <label htmlFor="category-filter">Category:</label>
              <select
                id="category-filter"
                value={localFilters.category || 'all'}
                onChange={(e) => handleInputChange('category', e.target.value)}
              >
                <option value="all">All Categories</option>
                {availableCategories.map(category => (
                  <option key={category} value={category}>
                    {category}
                  </option>
                ))}
              </select>
            </div>

            <div className="filter-group">
              <label htmlFor="impact-filter">Impact Level:</label>
              <select
                id="impact-filter"
                value={localFilters.impact || 'all'}
                onChange={(e) => handleInputChange('impact', e.target.value)}
              >
                <option value="all">All Impact Levels</option>
                {availableImpacts.map(impact => (
                  <option key={impact} value={impact}>
                    {impact}
                  </option>
                ))}
              </select>
            </div>
          </div>
        </div>

        {/* Price Range Filters */}
        <div className="filter-section">
          <h4>Price Range</h4>
          <div className="filter-row">
            <div className="filter-group">
              <label htmlFor="min-price">Min Price ($):</label>
              <input
                id="min-price"
                type="number"
                min="0"
                step="0.01"
                placeholder="0.00"
                value={localFilters.priceRange?.min || ''}
                onChange={(e) => handleInputChange('priceRange', {
                  ...localFilters.priceRange,
                  min: e.target.value
                })}
              />
            </div>

            <div className="filter-group">
              <label htmlFor="max-price">Max Price ($):</label>
              <input
                id="max-price"
                type="number"
                min="0"
                step="0.01"
                placeholder="200.00"
                value={localFilters.priceRange?.max || ''}
                onChange={(e) => handleInputChange('priceRange', {
                  ...localFilters.priceRange,
                  max: e.target.value
                })}
              />
            </div>
          </div>
        </div>

        {/* Action Buttons */}
        <div className="filter-actions">
          <button 
            className="filter-btn apply-btn"
            onClick={applyFilters}
            title="Apply current filter settings"
          >
            Apply Filters
          </button>
          
          <button 
            className="filter-btn reset-btn"
            onClick={resetFilters}
            title="Reset all filters to default values"
          >
            Reset Filters
          </button>
        </div>

        {/* Active Filters Summary */}
        <div className="active-filters">
          <h4>Active Filters:</h4>
          <div className="filter-tags">
            {localFilters.startDate && (
              <span className="filter-tag">
                From: {new Date(localFilters.startDate).toLocaleDateString()}
              </span>
            )}
            {localFilters.endDate && (
              <span className="filter-tag">
                To: {new Date(localFilters.endDate).toLocaleDateString()}
              </span>
            )}
            {localFilters.category && localFilters.category !== 'all' && (
              <span className="filter-tag">
                Category: {localFilters.category}
              </span>
            )}
            {localFilters.impact && localFilters.impact !== 'all' && (
              <span className="filter-tag">
                Impact: {localFilters.impact}
              </span>
            )}
            {localFilters.priceRange?.min && (
              <span className="filter-tag">
                Min Price: ${localFilters.priceRange.min}
              </span>
            )}
            {localFilters.priceRange?.max && (
              <span className="filter-tag">
                Max Price: ${localFilters.priceRange.max}
              </span>
            )}
          </div>
        </div>
      </div>
    </div>
  );
};

export default FilterPanel;