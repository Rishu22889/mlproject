from flask import Flask, render_template, request, jsonify
import numpy as np
import pickle
import os
import sys

from src.ml_project.logger import logging
from src.ml_project.exception import CustomException
from src.ml_project.pipelines.prediction_pipeline import PredictionPipeline

app = Flask(__name__)

# Initialize prediction pipeline
prediction_pipeline = PredictionPipeline()

@app.route('/')
def index():
    """Render the main page"""
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    """Handle predictions"""
    try:
        data = request.get_json()
        
        # Extract data
        passenger_data = {
            'Pclass': data.get('Pclass'),
            'Sex': data.get('Sex'),
            'Age': data.get('Age'),
            'Fare': data.get('Fare'),
            'SibSp': data.get('SibSp'),
            'Parch': data.get('Parch'),
            'Embarked': data.get('Embarked')
        }
        
        logging.info(f"Prediction request received: {passenger_data}")
        
        # Make prediction
        result = prediction_pipeline.predict(passenger_data)
        
        # Return result
        return jsonify({
            'survived': int(result[0]),
            'probability': float(result[1])
        })
        
    except Exception as e:
        logging.error(f"Error in prediction: {str(e)}")
        return jsonify({'error': str(e)}), 400

@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint"""
    return jsonify({'status': 'healthy'})

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=8000)
    