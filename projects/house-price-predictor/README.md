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

house-price-predictor/ ├── backend/ # Flask API and ML model │ ├── app.py # Main backend server │ ├── train_model.py # Model training script │ ├── house_model.pkl # Trained model │ └── scaler.pkl # Feature scaler │ └── frontend/ # React user interface ├── src/ └── package.json

## 🚀 How to Run Locally

cd backend python3 app.py

cd frontend npm start
