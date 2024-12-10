from flask import Flask, request, jsonify
import pickle
import numpy as np

# Initialize Flask app
app = Flask(__name__)

# Load the trained model and scaler
with open('./models/predictive_maintenance_model.pkl', 'rb') as f:
    model = pickle.load(f)

with open('./models/scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

@app.route('/predict', methods=['POST'])
def predict():
    """
    API endpoint for predicting RUL.
    Accepts a JSON POST request with sensor data and returns the predicted RUL.
    """
    data = request.json
    input_data = [data['temperature'], data['vibration'], data['pressure'], data['vibration_change'], data['temp_change']]
    input_data = np.array(input_data).reshape(1, -1)
    input_data = scaler.transform(input_data)  # Scale input
    prediction = model.predict(input_data)
    response = {'RUL': prediction[0]}
    return jsonify(response)

if __name__ == '__main__':
    app.run(debug=True)
