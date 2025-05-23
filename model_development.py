import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import joblib
from cslib import convert_to_ts

def engineer_features(df, training=True):
    """Create time-series features and target for revenue prediction."""
    # Sort by date
    df = df.sort_values('date')
    
    # Engineer features: lagged revenue
    df['prev_day_revenue'] = df['revenue'].shift(1)
    df['prev_week_revenue'] = df['revenue'].shift(7)
    df['prev_month_revenue'] = df['revenue'].shift(30)
    
    # Create target: next day's revenue
    df['target_revenue'] = df['revenue'].shift(-1)
    
    # Fill NaNs with 0
    df = df.fillna(0)
    
    if training:
        # Remove rows where target is NaN (last row)
        df = df.dropna(subset=['target_revenue'])
    
    return df

# Set up directories
data_dir = "E:/My-AI-workflow-capstone/data"
csv_file = os.path.join(data_dir, "data.csv")
os.makedirs("models", exist_ok=True)
os.makedirs("figures", exist_ok=True)

# Load and process data
df_raw = pd.read_csv(csv_file)
df_raw['invoice_date'] = pd.to_datetime(df_raw['invoice_date'])
df_ts = convert_to_ts(df_raw, country=None)  # Aggregate to daily time-series

# Engineer features
feature_matrix = engineer_features(df_ts, training=True)
feature_matrix.to_csv("data/feature_matrix.csv", index=False)

# Split data
# If only one day, use all data for training (adjust when more data is available)
if len(df_ts) <= 1:
    print("Warning: Only one day of data. Using all for training. Add more data for testing.")
    train = feature_matrix
    test = pd.DataFrame()  # Empty test set
else:
    # Use last day for testing (adjust to 30 days with more data)
    train = feature_matrix[feature_matrix['date'] < feature_matrix['date'].max()]
    test = feature_matrix[feature_matrix['date'] >= feature_matrix['date'].max()]

# Prepare features and target
X_train = train[['prev_day_revenue', 'prev_week_revenue', 'prev_month_revenue']]
y_train = train['target_revenue']

# Train models
if not test.empty:
    X_test = test[['prev_day_revenue', 'prev_week_revenue', 'prev_month_revenue']]
    y_test = test['target_revenue']
    
    # Baseline: Mean revenue
    baseline_pred = np.full_like(y_test, y_train.mean())
    baseline_mse = mean_squared_error(y_test, baseline_pred)
    
    # Linear Regression
    lr_model = LinearRegression()
    lr_model.fit(X_train, y_train)
    lr_pred = lr_model.predict(X_test)
    lr_mse = mean_squared_error(y_test, lr_pred)
    
    # Random Forest
    rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
    rf_model.fit(X_train, y_train)
    rf_pred = rf_model.predict(X_test)
    rf_mse = mean_squared_error(y_test, rf_pred)
    
    # Save best model (Random Forest, assuming lowest MSE)
    joblib.dump(rf_model, 'models/rf_model.pkl')
    
    # Visualize comparison
    plt.figure(figsize=(8, 5))
    plt.bar(['Baseline', 'Linear Regression', 'Random Forest'], 
            [baseline_mse, lr_mse, rf_mse], 
            color=['#FF6384', '#36A2EB', '#FFCE56'])
    plt.title('Model Performance Comparison')
    plt.ylabel('Mean Squared Error')
    plt.tight_layout()
    plt.savefig('figures/model_comparison.png')
    plt.close()
else:
    # Train Random Forest on all data if no test set
    rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
    rf_model.fit(X_train, y_train)
    joblib.dump(rf_model, 'models/rf_model.pkl')
    print("No test data. Model trained and saved without evaluation.")

plt.figure(figsize=(10, 6))
plt.bar(['Baseline', 'Linear Regression', 'Random Forest'], [baseline_mse, lr_mse, rf_mse])
plt.title('Model Comparison: Mean Squared Error')
plt.ylabel('MSE')
plt.savefig('model_comparison.png')
plt.close()

# Unit tests
import pytest

def test_model_load():
    model = joblib.load('models/rf_model.pkl')
    assert model is not None

def test_model_predict():
    model = joblib.load('models/rf_model.pkl')
    sample_input = np.array([[100, 500, 2000]])
    pred = model.predict(sample_input)
    assert len(pred) == 1

print("Model training complete. Run 'pytest' to execute unit tests.")