from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import numpy as np

app = Flask(__name__)
CORS(app)

model = joblib.load('C:\Codes\heart_api\custom_heart_disease_model.pkl')
scaler = joblib.load('C:\Codes\heart_api\scaler.pkl')  # Remove if no scaler

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        features = [
            data['age'],
            data['sex'],
            data['cp'],
            data['trestbps'],
            data['chol'],
            data['thalach'],
            data['exang'],
            data['oldpeak'],
            data['ca'],
            data['thal']
        ]
        features = np.array(features).reshape(1, -1)
        features = scaler.transform(features)  # Remove if no scaler
        prediction = model.predict(features)[0]
        return jsonify({'prediction': int(prediction)})
    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000)