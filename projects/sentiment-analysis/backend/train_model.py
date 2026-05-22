# For working with arrays and numerical operations
import numpy as np
# Converts text into numerical features
from sklearn.feature_extraction.text import TfidfVectorizer
# The machine learning model (ensemble of decision trees)
from sklearn.ensemble import RandomForestClassifier
# Splits data into training and testing sets
from sklearn.model_selection import train_test_split
# Used to save and load trained models to/from files
import joblib

print("=== Training Strong & Balanced Sentiment Model ===")

# Large, diverse, and balanced dataset
texts = [
    # Positive examples (20)
    "I love this product, it's amazing!", "Absolutely fantastic experience",
    "Very happy with my purchase", "Best decision I made this year",
    "This exceeded my expectations", "Outstanding quality and service",
    "I would buy this again in a heartbeat", "Great value for money",
    "Superb product, highly recommend", "Changed my life for the better",
    "Perfect in every way", "Wonderful experience", "Best purchase ever",
    "I'm extremely satisfied", "This is excellent", "Top quality product",
    "Highly recommend to everyone", "Works perfectly", "Very impressed",
    "Worth every penny",

    # Negative examples (20)
    "This is the worst thing I ever bought", "Terrible quality",
    "Complete waste of money", "Awful service and product",
    "Very disappointed with this purchase", "Broke after one use",
    "Worst product ever", "Total garbage", "Extremely poor quality",
    "Regret buying this", "Doesn't work at all", "Customer service is horrible",
    "Cheap and breaks easily", "Avoid at all costs", "Big disappointment",
    "Complete failure", "Useless product", "Never buying again",
    "Horrible experience", "Waste of time and money"
]

# Perfectly balanced labels: 1 = Positive, 0 = Negative
sentiments = [1] * 20 + [0] * 20

# Convert text data into numerical features using TF-IDF
vectorizer = TfidfVectorizer(
    max_features=5000,                 # Keep only the 5000 most important words
    stop_words='english',              # Remove common words like "the", "is", "and"
    # Consider single words, pairs, and triplets of words
    ngram_range=(1, 3)
)
# Transform text into numerical matrix
X = vectorizer.fit_transform(texts).toarray()

# Convert sentiment list into numpy array
y = np.array(sentiments)

# Split the data into training set (70%) and testing set (30%)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42)

# Create the Random Forest model with specific settings
model = RandomForestClassifier(
    n_estimators=500,                  # Number of decision trees in the forest
    # Maximum depth of each tree (prevents overfitting)
    max_depth=12,
    # Automatically balance positive and negative classes
    class_weight='balanced',
    random_state=42                    # For reproducible results
)
model.fit(X_train, y_train)            # Train the model on the training data

# Evaluate how well the model performs on unseen test
