# app.py
# This is the backend server that receives text from React and returns sentiment prediction

# Flask framework for creating API
from flask import Flask, request, jsonify
# Allows React frontend to connect
from flask_cors import CORS
import joblib                                  # To load saved model

app = Flask(__name__)                          # Create Flask application
# Enable Cross-Origin requests from React
CORS(app)

# Load the trained model and vectorizer from files
model = joblib.load('sentiment_model.pkl')
vectorizer = joblib.load('vectorizer.pkl')


# Define API endpoint for predictions
@app.route('/predict', methods=['POST'])
def predict():
    data = request.json                        # Get JSON data sent from React
    text = data.get('text', '')                # Extract the text from request

    if not text:                               # If no text was provided
        return jsonify({'error': 'No text provided'}), 400   # Return error

    # Convert text to numbers and predict
    vectorized = vectorizer.transform([text]).toarray()
    prediction = model.predict(vectorized)[0]   # Get prediction (0 or 1)

    sentiment = "Positive" if prediction == 1 else "Negative"   # Convert to readable text

    return jsonify({                           # Send response back to React
        'sentiment': sentiment,
        'text': text
    })


@app.route('/')                                # Home route
def home():
    return "Sentiment Analysis API is running! 🚀"


if __name__ == '__main__':                     # Run the server when file is executed
    # debug=True shows errors clearly
    app.run(debug=True)
