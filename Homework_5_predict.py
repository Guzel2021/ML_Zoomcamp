
import pickle
import numpy as np

from flask import Flask, request, jsonify

def predict_single(customer, dv, model):
    X = dv.transform([customer])
    y_pred = model.predict_proba(X)[:, 1]
    return y_pred[0]


# Load the DictVectorizer from dv.bin
with open('dv.bin', 'rb') as f_in:
    dv = pickle.load(f_in)

# Load the model from model1.bin
with open('model1.bin', 'rb') as f_in:
    model = pickle.load(f_in)


app = Flask('decision')


@app.route('/predict', methods=['POST'])
def predict():
    customer = request.get_json()

    prediction = predict_single(customer, dv, model)
    decision = prediction >= 0.5
    
    result = {
        'decision_probability': float(prediction),
        'decision': bool(decision),
    }

    return jsonify(result)

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=9696)


