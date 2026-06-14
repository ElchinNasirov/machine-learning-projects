# House Price Predictor

A full-stack web application that predicts house prices based on size, number of bedrooms, and age of the property.

## ✨ Features

- Real-time price prediction
- Clean and responsive React interface
- Flask REST API backend
- Linear Regression model with feature scaling

## 🛠 Tech Stack

- **Frontend**: React
- **Backend**: Flask + Flask-CORS
- **Machine Learning**: LinearRegression (Scikit-Learn)
- **Data Processing**: StandardScaler

## 📁 Project Structure

house-price-predictor/
├── backend/                    # Flask API and ML model
│   ├── app.py                  # Main backend server
│   ├── train_model.py          # Model training script
│   ├── house_model.pkl         # Trained model
│   └── scaler.pkl              # Feature scaler
│
└── frontend/                   # React user interface
    ├── src/
    └── package.json


## 🚀 How to Run Locally

**Backend:**

cd backend

python3 app.py

##

**Frontend:**

cd frontend

npm start

##

The React app will open at http://localhost:3000 and connect to the Flask backend at http://127.0.0.1:5000
