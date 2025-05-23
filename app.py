# app.py
from flask import Flask, jsonify, request
import pandas as pd
import joblib
import logging
import os
from scipy.stats import wasserstein_distance

app = Flask(__name__)

os.makedirs('logs', exist_ok=True)
logging.basicConfig(filename='logs/app.log', level=logging.INFO, 
                    format='%(asctime)s - %(levelname)s - %(message)s')

# Load model and feature matrix
model = joblib.load('models/rf_model.pkl')
feature_matrix = pd.read_csv('data/feature_matrix.csv')
feature_matrix['date'] = pd.to_datetime(feature_matrix['date'])

def monitor_performance(actual, predicted):
    """Calculate Wasserstein distance for performance monitoring."""
    return wasserstein_distance([actual], [predicted]) if actual is not None else float('inf')

@app.route('/predict', methods=['POST'])
def predict():
    """Predict next-day revenue for a given date."""
    if not request.json or 'query' not in request.json:
        logging.error("Invalid request: Missing query")
        return jsonify({'error': 'Invalid request: Missing query'}), 400
    
    query = request.json['query']
    date = query.get('date')  # Expected format: YYYY-MM-DD
    
    try:
        date = pd.to_datetime(date)
    except ValueError:
        logging.error(f"Invalid date format: {date}")
        return jsonify({'error': 'Invalid date format, use YYYY-MM-DD'}), 400
    
    # Find features for the given date
    features = feature_matrix[feature_matrix['date'] == date][['prev_day_revenue', 'prev_week_revenue', 'prev_month_revenue']]
    
    if features.empty:
        available_dates = feature_matrix['date'].dt.strftime('%Y-%m-%d').unique().tolist()
        logging.warning(f"No data for date {date}")
        return jsonify({'error': f'No data for date {date}', 'available_dates': available_dates}), 404
    
    # Predict
    prediction = model.predict(features)[0]
    
    # Log prediction
    logging.info(f"Prediction for {date}: {prediction}")
    
    # Monitor
    actual = feature_matrix[feature_matrix['date'] == date + pd.Timedelta(days=1)]['target_revenue'].values
    actual = actual[0] if len(actual) > 0 else None
    distance = monitor_performance(actual, prediction)
    logging.info(f"Monitoring: Wasserstein distance = {distance}")
    
    return jsonify({
        'date': str(date.date()),
        'predicted_revenue': float(prediction),
        'monitoring_distance': float(distance)
    })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080)