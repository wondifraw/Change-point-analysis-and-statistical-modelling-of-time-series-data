# Task 3: Interactive Dashboard for Oil Price Analysis

## Overview
Comprehensive Flask + React dashboard integrating analysis results from Tasks 1 & 2 with interactive visualizations and filtering capabilities.

## Architecture
```
Task 3 Dashboard
├── Backend (Flask)
│   ├── Integrates Task 1 (Event Compiler, Change Point Model)
│   ├── Integrates Task 2 (Bayesian Change Point Analysis)
│   ├── RESTful APIs with error handling
│   └── Data caching and optimization
└── Frontend (React)
    ├── Interactive price charts with Recharts
    ├── Filterable events table
    ├── Change points visualization
    ├── Summary cards with key metrics
    └── Responsive design for all devices
```

## Key Features

### Backend APIs
- **GET /api/health** - Health check and system status
- **GET /api/oil-prices** - Oil price time series data
- **GET /api/events** - Geopolitical events from Task 1
- **GET /api/change-points** - Change points from Task 2 analysis
- **GET /api/analysis-summary** - Comprehensive statistics
- **GET /api/event-impact-analysis** - Event-price correlation analysis

### Frontend Components
- **PriceChart** - Interactive timeline with events overlay
- **ChangePointsChart** - Bayesian analysis results visualization
- **EventsTable** - Sortable/filterable events display
- **FilterPanel** - Date range, category, and impact filters
- **SummaryCards** - Key metrics and insights dashboard

## Installation & Setup

### Backend Setup
```bash
cd src/task3/backend
pip install -r requirements.txt
python app.py
```

### Frontend Setup
```bash
cd src/task3/frontend
npm install
npm start
```

## Integration with Tasks 1 & 2

### Task 1 Integration
- Reuses `EventCompiler` for geopolitical events
- Leverages `ChangePointModel` for structural break detection
- Imports `DataWorkflow` for data processing pipeline

### Task 2 Integration
- Incorporates Bayesian change point results
- Displays confidence intervals and impact metrics
- Shows hypothesis generation from event associations

## Error Handling
- Comprehensive try-catch blocks in all functions
- Graceful degradation when modules unavailable
- User-friendly error messages and troubleshooting tips
- Backend health checks and status monitoring

## Usage Examples

### Start the Full Stack
```bash
# Terminal 1: Start Flask backend
cd src/task3/backend
python app.py

# Terminal 2: Start React frontend
cd src/task3/frontend
npm start
```

### Access Dashboard
- Open browser to `http://localhost:3000`
- Backend API available at `http://localhost:5000`

## API Response Examples

### Oil Prices Endpoint
```json
{
  "success": true,
  "data": [
    {
      "date": "2022-02-24",
      "price": 105.78,
      "year": 2022,
      "month": 2
    }
  ],
  "count": 9247,
  "date_range": {
    "start": "1987-05-20",
    "end": "2022-11-14"
  }
}
```

### Change Points Endpoint
```json
{
  "success": true,
  "data": [
    {
      "date": "2008-09-15",
      "confidence": 0.95,
      "before_mean": 97.5,
      "after_mean": 45.2,
      "change_percent": -53.6,
      "method": "PELT"
    }
  ],
  "count": 3
}
```

## Responsive Design
- Mobile-first approach
- Tablet and desktop optimizations
- Touch-friendly interactions
- Accessible design patterns

## Performance Optimizations
- Data caching in backend
- Lazy loading of components
- Efficient chart rendering
- Debounced filter updates

## Dependencies

### Backend
- Flask 2.3.3
- Flask-CORS 4.0.0
- pandas 2.0.3
- numpy 1.24.3

### Frontend
- React 18.2.0
- Recharts 2.8.0
- Axios 1.5.0

## Troubleshooting

### Common Issues
1. **Backend not starting**: Check if Tasks 1 & 2 modules are installed
2. **No data loading**: Verify data files exist in `data/raw/`
3. **CORS errors**: Ensure Flask-CORS is installed and configured
4. **Chart not rendering**: Check if data format matches expected structure

### Debug Mode
- Backend: Set `debug=True` in `app.run()`
- Frontend: Check browser console for React errors
- Network: Use browser dev tools to inspect API calls