#!/usr/bin/env python3
"""
Dzud Risk API - Flask Web Service
"""

from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from risk_predictor import DzudRiskPredictor
from daily_forecast_predictor import DailyDzudForecast
from historical_daily_data import HistoricalDailyData
import json

app = Flask(__name__)
CORS(app)  # Enable CORS for frontend

# Initialize predictor
predictor = DzudRiskPredictor()
forecaster = DailyDzudForecast()
historical = HistoricalDailyData()

@app.route('/')
def index():
    """Home page"""
    return render_template('index.html')

@app.route('/api/predict', methods=['POST'])
def predict():
    """
    Predict dzud risk
    
    Request body:
    {
        "lat": 43.5,
        "lon": 104.4,
        "livestock": {
            "sheep": 200,
            "goat": 150,
            "cattle": 50,
            "horse": 30,
            "camel": 10
        },
        "month": 1  (optional, defaults to current month)
    }
    """
    try:
        data = request.get_json()
        
        # Validate input
        if 'lat' not in data or 'lon' not in data:
            return jsonify({'error': 'lat and lon are required'}), 400
        
        if 'livestock' not in data:
            return jsonify({'error': 'livestock data is required'}), 400
        
        lat = float(data['lat'])
        lon = float(data['lon'])
        livestock = data['livestock']
        month = data.get('month', None)
        
        # Predict
        result = predictor.predict(lat, lon, livestock, month)
        
        return jsonify(result)
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/locations', methods=['GET'])
def get_locations():
    """Get available locations"""
    locations = predictor.weather_data[['aimag', 'soum', 'lat', 'lon']].drop_duplicates()
    result = locations.to_dict('records')
    print(f"📍 Returning {len(result)} locations")
    for loc in result:
        print(f"   - {loc['soum']}")
    return jsonify(result)

@app.route('/api/risk-map', methods=['POST'])
def get_risk_map():
    """
    Get risk levels for all locations
    
    Request body:
    {
        "livestock": {...},
        "month": 1
    }
    """
    try:
        data = request.get_json()
        livestock = data.get('livestock', {})
        month = data.get('month', None)
        
        # Get all unique locations
        locations = predictor.weather_data[['aimag', 'soum', 'lat', 'lon']].drop_duplicates()
        
        # Calculate risk for each location
        risk_map = []
        for _, loc in locations.iterrows():
            result = predictor.predict(loc['lat'], loc['lon'], livestock, month)
            risk_map.append({
                'soum': loc['soum'],
                'lat': loc['lat'],
                'lon': loc['lon'],
                'risk_score': result['risk']['score'],
                'risk_level': result['risk']['level'],
                'risk_label': result['risk']['label'],
                'color': result['risk']['color']
            })
        
        return jsonify(risk_map)
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/geojson', methods=['GET'])
def get_geojson():
    """Get Omnogovi soum boundaries from GeoJSON"""
    try:
        import json
        
        # Read simplified Omnogovi GeoJSON
        with open('omnogovi_soums_simple.geojson', 'r', encoding='utf-8') as f:
            geojson_data = json.load(f)
        
        return jsonify(geojson_data)
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/health', methods=['GET'])
def health():
    """Health check"""
    return jsonify({
        'status': 'ok',
        'model_loaded': predictor.has_model,
        'weather_data_rows': len(predictor.weather_data)
    })

@app.route('/api/forecast', methods=['POST'])
def get_forecast():
    """
    Get daily dzud risk forecast for next 7-14 days
    
    Request body:
    {
        "lat": 43.5,
        "lon": 104.4,
        "livestock": 200,
        "days": 14  (optional, default 14)
    }
    """
    try:
        data = request.get_json()
        
        lat = float(data.get('lat', 43.57))
        lon = float(data.get('lon', 104.43))
        livestock = float(data.get('livestock', 200))
        days = int(data.get('days', 14))
        
        # Get forecast
        results_df = forecaster.forecast(lat, lon, livestock, days, verbose=False)
        
        if results_df is None:
            return jsonify({'error': 'Failed to get weather forecast'}), 500
        
        # Convert to JSON-friendly format
        forecast_data = []
        for _, row in results_df.iterrows():
            forecast_data.append({
                'date': row['date'].strftime('%Y-%m-%d'),
                'day_name': row['day_name'],
                'temp_min': round(row['temp_min'], 1),
                'temp_max': round(row['temp_max'], 1),
                'temp_mean': round(row['temp_mean'], 1),
                'wind_max': round(row['wind_max'], 1),
                'snowfall': round(row['snowfall'], 1),
                'precip': round(row['precip'], 1),
                'risk_score': round(row['risk_score'], 0),
                'risk_level': int(row['risk_level']),
                'risk_label': row['risk_label'],
                'risk_color': row['risk_color'],
                'reasons': row['reasons']
            })
        
        # Summary
        high_risk_days = len(results_df[results_df['risk_level'] >= 2])
        coldest_day = results_df.loc[results_df['temp_min'].idxmin()]
        windiest_day = results_df.loc[results_df['wind_max'].idxmax()]
        
        summary = {
            'total_days': len(results_df),
            'high_risk_days': high_risk_days,
            'coldest_day': {
                'date': coldest_day['date'].strftime('%Y-%m-%d'),
                'temp': round(coldest_day['temp_min'], 1)
            },
            'windiest_day': {
                'date': windiest_day['date'].strftime('%Y-%m-%d'),
                'wind': round(windiest_day['wind_max'], 1)
            }
        }
        
        return jsonify({
            'forecast': forecast_data,
            'summary': summary
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/historical', methods=['POST'])
def get_historical():
    """
    Get historical daily dzud risk data
    
    Request body:
    {
        "lat": 43.5,
        "lon": 104.4,
        "start_date": "2023-01-01",
        "end_date": "2023-01-31",
        "livestock": 200  (optional)
    }
    """
    try:
        data = request.get_json()
        
        lat = float(data.get('lat', 43.57))
        lon = float(data.get('lon', 104.43))
        start_date = data.get('start_date', '2023-01-01')
        end_date = data.get('end_date', '2023-01-31')
        livestock = float(data.get('livestock', 200))
        
        # Get historical data
        daily_df = historical.get_daily_data(lat, lon, start_date, end_date, livestock)
        
        if len(daily_df) == 0:
            return jsonify({'error': 'No data available for this date range'}), 404
        
        # Convert to JSON
        historical_data = []
        for _, row in daily_df.iterrows():
            historical_data.append({
                'date': row['date'],
                'day_name': row['day_name'],
                'soum': row['soum'],
                'temp_min': float(row['temp_min']),
                'temp_avg': float(row['temp_avg']),
                'wind_speed': float(row['wind_speed']),
                'snowfall': float(row['snowfall']),
                'precip': float(row['precip']),
                'risk_score': int(row['risk_score']),
                'risk_level': int(row['risk_level']),
                'risk_label': row['risk_label'],
                'risk_color': row['risk_color']
            })
        
        # Summary
        high_risk_days = int(len(daily_df[daily_df['risk_level'] >= 2]))
        summary = {
            'total_days': int(len(daily_df)),
            'avg_risk_score': float(round(daily_df['risk_score'].mean(), 1)),
            'max_risk_score': int(round(daily_df['risk_score'].max(), 0)),
            'high_risk_days': high_risk_days,
            'soum': str(daily_df['soum'].iloc[0])
        }
        
        if len(daily_df) > 0:
            summary['coldest_day'] = {
                'date': str(daily_df.loc[daily_df['temp_min'].idxmin(), 'date']),
                'temp': float(round(daily_df['temp_min'].min(), 1))
            }
            summary['windiest_day'] = {
                'date': str(daily_df.loc[daily_df['wind_speed'].idxmax(), 'date']),
                'wind': float(round(daily_df['wind_speed'].max(), 1))
            }
        
        return jsonify({
            'historical': historical_data,
            'summary': summary
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    print("🚀 Starting Dzud Risk API...")
    print("📍 API endpoint: http://localhost:5001/api/predict")
    print("🌐 Web interface: http://localhost:5001/")
    app.run(debug=True, host='0.0.0.0', port=5001)
