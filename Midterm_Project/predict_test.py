import requests

# Define the URL of your Flask server
url = "http://127.0.0.1:9696/predict"

# Define the payload (example student data)
data = {
    "school": "GP",
    "sex": "F",
    "age": 15,
    "address": "U",
    "famsize": "GT3",
    "pstatus": "T",
    "medu": 3,
    "fedu": 1,
    "mjob": "at_home",
    "fjob": "services",
    "reason": "reputation",
    "guardian": "mother",
    "traveltime": 1,
    "studytime": 2,
    "failures": 0,
    "schoolsup": "no",
    "famsup": "yes",
    "paid": "no",
    "activities": "no",
    "nursery": "yes",
    "higher": "yes",
    "internet": "yes",
    "romantic": "no",
    "famrel": 4,
    "freetime": 3,
    "goout": 3,
    "dalc": 1,
    "walc": 1,
    "health": 5,
    "absences": 0,
    "g1": 10,
    "g2": 11
}

# Send the POST request
response = requests.post(url, json=data)

# Print the response
if response.status_code == 200:
    print("Prediction response:", response.json())
else:
    print("Error:", response.status_code, response.text)
