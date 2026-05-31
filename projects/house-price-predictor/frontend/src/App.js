import React, { useState } from 'react';           // Import React and useState for managing state
import './App.css';                               // Import CSS file for styling

function App() {                                  // Main React component
  const [formData, setFormData] = useState({      // State to hold house details
    size: '',
    bedrooms: '',
    age: ''
  });
  const [result, setResult] = useState(null);     // State to store prediction result
  const [loading, setLoading] = useState(false);  // State for loading indicator

  // Handle input changes for all fields
  const handleChange = (e) => {
    setFormData({
      ...formData,
      [e.target.name]: e.target.value
    });
  };

  // Send data to Flask backend for prediction
  const predictPrice = async () => {
    if (!formData.size || !formData.bedrooms || !formData.age) {
      alert("Please fill all fields");
      return;
    }

    setLoading(true);

    try {
      const response = await fetch('http://127.0.0.1:5000/predict', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(formData)
      });

      const data = await response.json();
      setResult(data);
    } catch (error) {
      console.error(error);
      alert("Error connecting to backend. Make sure Flask is running on port 5000.");
    }

    setLoading(false);
  };

  return (
    <div className="App">
      <div className="container">
        <h1>🏠 House Price Predictor</h1>
        <p>Enter house details to get estimated price in USD</p>

        <div className="form-group">
          <label>Size (sq ft)</label>
          <input
            type="number"
            name="size"
            value={formData.size}
            onChange={handleChange}
            placeholder="e.g. 2500"
          />
        </div>

        <div className="form-group">
          <label>Bedrooms</label>
          <input
            type="number"
            name="bedrooms"
            value={formData.bedrooms}
            onChange={handleChange}
            placeholder="e.g. 4"
          />
        </div>

        <div className="form-group">
          <label>Age (years)</label>
          <input
            type="number"
            name="age"
            value={formData.age}
            onChange={handleChange}
            placeholder="e.g. 10"
          />
        </div>

        <button
          onClick={predictPrice}
          disabled={loading}
          className="predict-btn"
        >
          {loading ? 'Predicting...' : 'Predict Price'}
        </button>

        {result && (
          <div className="result-card">
            <h2>Estimated Price</h2>
            <h1>${result.predicted_price.toLocaleString()}</h1>
            <p>
              Size: {result.features.size} sq ft |
              Bedrooms: {result.features.bedrooms} |
              Age: {result.features.age} years
            </p>
          </div>
        )}
      </div>
    </div>
  );
}

export default App;