# app.py - House Price Predictor Flask Backend

# Flask for API, jsonify for JSON responses
from flask import Flask, request, jsonify
# Allow React to connect to backend
from flask_cors import CORS
import joblib                                    # Load saved model and scaler
import numpy as np                               # For handling input data

app = Flask(__name__)                            # Create Flask application
# Enable CORS for React frontend
CORS(app)

# Load the trained model and scaler
model = joblib.load('house_model.pkl')           # Load Linear Regression model
# Load the scaler used during training
scaler = joblib.load('scaler.pkl')


@app.route('/predict', methods=['POST'])         # API endpoint for predictions
def predict():
    data = request.json                          # Get JSON data from React

    # Extract features from request
    size = data.get('size')
    bedrooms = data.get('bedrooms')
    age = data.get('age')

    # Prepare features as array
    features = np.array([[size, bedrooms, age]])

    # Scale the features (same as during training)
    scaled_features = scaler.transform(features)

    # Make prediction
    prediction = model.predict(scaled_features)[0]

    return jsonify({                             # Return prediction as JSON
        'predicted_price': round(float(prediction), 2),
        'features': {
            'size': size,
            'bedrooms': bedrooms,
            'age': age
        }
    })


@app.route('/')                                  # Root route for testing
def home():
    return "House Price Predictor API is running! 🚀"


if __name__ == '__main__':                       # Run the server when file is executed
    app.run(debug=True)                          # Debug mode for development
