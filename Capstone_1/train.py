#!/usr/bin/env python
# coding: utf-8
import pandas as pd
from sklearn.feature_extraction import DictVectorizer
import xgboost as xgb
import pickle
from flask import Flask, request, jsonify
import logging

# Load the model and DictVectorizer
input_file = 'model.bin'
with open(input_file, 'rb') as f:
    dv, model = pickle.load(f)

# Initialize the Flask app
app = Flask('predict')

# Retrieve features from the DictVectorizer
try:
    features = list(dv.get_feature_names_out())
except AttributeError:
    features = list(dv.feature_names)

def predict_single(customer, dv, model):
    # Fill missing features with default values
    for feature in features:
        if feature not in customer:
            customer[feature] = 0  # Default value for missing features

    # Transform customer data
    customer_dict = [customer]
    customer_transformed = dv.transform(customer_dict)

    # Convert to XGBoost DMatrix
    customer_dmatrix = xgb.DMatrix(customer_transformed, feature_names=features)

    # Predict
    customer_prediction = model.predict(customer_dmatrix)
    
    # Convert the probability into a binary classification
    threshold = 0.5
    customer_class = 1 if customer_prediction[0] >= threshold else 0

    return customer_prediction[0], customer_class

# Define Flask endpoint
@app.route('/predict', methods=['POST'])
def predict_endpoint():
    customer = request.get_json()

    try:
        # Unpack the tuple returned by predict_single
        probability, predicted_class = predict_single(customer, dv, model)
        
        # Return both probability and binary class in the response
        result = {
            'probability': float(probability),  # Single prediction probability
            'class': int(predicted_class)       # Binary class (1 or 0)
        }
        return jsonify(result)
    except Exception as e:
        logging.error(f"Error during prediction: {str(e)}")
        return jsonify({'error': str(e)}), 400

# Main entry point for development
if __name__ == '__main__':
    from waitress import serve
    print("Starting the server...")
    serve(app, host='0.0.0.0', port=9696)
