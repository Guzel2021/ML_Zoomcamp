import requests
import pickle
import json
import xgboost as xgb

# URL of the Flask API
url = "http://127.0.0.1:9696/predict"

# Load the DictVectorizer and the model from the pickle file
with open('model.bin', 'rb') as f_in:
    dv, model = pickle.load(f_in)


# Customer data (the input data you want to predict for)
customer = {
    "person_age": 26.0, 
    "person_gender": "F", 
    "person_education": "Master", 
    "person_income": 99999.0, 
    "person_emp_exp": 10, 
    "person_home_ownership": "MORTGAGE", 
    "loan_amnt": 25000.0, 
    "loan_intent": "EDUCATION", 
    "loan_int_rate": 15.77, 
    "loan_percent_income": 0.15, 
    "cb_person_cred_hist_length": 3.0, 
    "credit_score": 584, 
    "previous_loan_defaults_on_file": "No"
}

# Send the POST request
response = requests.post(url, json=customer)

# Print the response
if response.status_code == 200:
    print("Prediction response:", response.json())
else:
    print("Error:", response.status_code, response.text)