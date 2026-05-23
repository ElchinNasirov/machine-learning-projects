# This is the backend server that receives text from React and returns sentiment prediction

# Import Flask for creating the web server and jsonify for sending JSON responses
from flask import Flask, request, jsonify
# Import CORS to allow React frontend to connect to this backend
from flask_cors import CORS
# Import joblib to load the saved machine learning model
import joblib

# Create a new Flask web application instance
app = Flask(__name__)
# Enable Cross-Origin Resource Sharing so React can communicate with Flask
CORS(app)

# Load the trained model and vectorizer from saved files
model = joblib.load('sentiment_model.pkl')       # Load the RandomForest model
# Load the TF-IDF vectorizer used to convert text to numbers
vectorizer = joblib.load('vectorizer.pkl')


# Define a route for POST requests to /predict
@app.route('/predict', methods=['POST'])
def predict():
    # Get the JSON data sent from the React frontend
    data = request.json
    # Extract the 'text' field from the request (default to empty string if missing)
    text = data.get('text', '')

    if not text:                                 # If no text was provided
        # Return error response with status code 400
        return jsonify({'error': 'No text provided'}), 400

    # Convert input text into numerical features using the saved vectorizer
    vectorized = vectorizer.transform([text]).toarray()

    # Make prediction using the loaded model
    prediction = model.predict(vectorized)[0]    # Get the prediction (0 or 1)

    # Get confidence score (probability of the predicted class)
    probabilities = model.predict_proba(
        vectorized)[0]   # Get probability for each class
    # Take the highest probability as confidence
    confidence = max(probabilities)

    # Convert numeric prediction to readable text
    sentiment = "Positive" if prediction == 1 else "Negative"

    # Return the result as JSON
    return jsonify({
        'sentiment': sentiment,                  # "Positive" or "Negative"
        # Confidence score (e.g., 0.8765)
        'confidence': round(confidence, 4),
        'text': text                             # Return the original text for display
    })


# Root route for basic health check
@app.route('/')
def home():
    # Simple message when accessing the root URL
    return "Sentiment Analysis API is running! 🚀"


if __name__ == '__main__':                       # This block runs only when the file is executed directly
    # Start the Flask development server with debug mode enabled
    app.run(debug=True)
