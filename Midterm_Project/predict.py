#!/usr/bin/env python
# coding: utf-8
import pandas as pd
from sklearn.feature_extraction import DictVectorizer
import xgboost as xgb
import pickle
from flask import Flask, request, jsonify

# File with the saved model and DictVectorizer
input_file = 'model.bin'

# Load the DictVectorizer and model
with open(input_file, 'rb') as f_in: 
    dv, model = pickle.load(f_in)

# Initialize the Flask app
app = Flask('performance')

# Get trained features
trained_features = list(dv.get_feature_names_out())
print("Model was trained with the following features:")
print(trained_features)

# Prediction function
def predict_performance(data, model, dv):
    # Convert input data to DataFrame
    data_df = pd.DataFrame([data])  # Single-row DataFrame

    # Transform using DictVectorizer
    data_dict = data_df.to_dict(orient='records')
    X = dv.transform(data_dict)

    # Ensure feature names are passed correctly
    if len(trained_features) != X.shape[1]:
        raise ValueError("Mismatch between features and transformed data dimensions")

    # Convert to DMatrix for XGBoost
    dmatrix = xgb.DMatrix(X, feature_names=trained_features)
    return model.predict(dmatrix)


# Define Flask endpoint
@app.route('/predict', methods=['POST'])
def predict_endpoint():
    student = request.get_json()

    try:
        prediction = predict_performance(student, model, dv)
        result = {
            'performance': float(prediction[0]),  # Assuming single prediction
        }
        return jsonify(result)
    except Exception as e:
        return jsonify({'error': str(e)}), 400


# Main entry point for development
if __name__ == '__main__':
    from waitress import serve
    print("Starting the server...")
    serve(app, host='0.0.0.0', port=9696)


