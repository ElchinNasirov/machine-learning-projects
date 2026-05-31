# train_model.py - House Price Predictor
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import joblib

print("=== Training House Price Predictor Model (Simplified) ===\n")

# Data using NumPy arrays
sizes = np.array([1200, 1500, 1800, 2000, 2500, 2800, 3000, 3200, 3500, 4000])
bedrooms = np.array([2, 3, 3, 4, 4, 5, 5, 4, 5, 6])
ages = np.array([10, 5, 15, 8, 20, 3, 12, 7, 25, 2])
prices = np.array([250000, 320000, 380000, 450000, 520000,
                  580000, 620000, 670000, 720000, 850000])

# Combine features
X = np.column_stack((sizes, bedrooms, ages))
y = prices

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42)

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train model
model = LinearRegression()
model.fit(X_train_scaled, y_train)

# Evaluate
y_pred = model.predict(X_test_scaled)
print("R² Score:", r2_score(y_test, y_pred))
print("Mean Squared Error:", mean_squared_error(y_test, y_pred))

# Save model and scaler
joblib.dump(model, 'house_model.pkl')
joblib.dump(scaler, 'scaler.pkl')

print("Model and scaler saved successfully!")
