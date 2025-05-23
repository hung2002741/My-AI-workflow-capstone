# test_api.py
import pytest
from app import app
import json

@pytest.fixture
def client():
    app.config['TESTING'] = True
    with app.test_client() as client:
        yield client

def test_predict_endpoint_valid(client):
    response = client.post('/predict', 
                         data=json.dumps({'query': {'date': '2017-11-28', 'country': 'United Kingdom'}}),
                         content_type='application/json')
    assert response.status_code == 200
    assert 'predicted_revenue' in response.json

def test_predict_endpoint_invalid_date(client):
    response = client.post('/predict', 
                         data=json.dumps({'query': {'date': 'invalid-date', 'country': 'United Kingdom'}}),
                         content_type='application/json')
    assert response.status_code == 400
    assert 'error' in response.json