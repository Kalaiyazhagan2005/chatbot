# intent_model.py
# KalaiBot - Step 1: Intent Classifier using TF-IDF + Logistic Regression

import json
import random
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import joblib

# Sample dataset: intents and example patterns
intents = {
    "greeting": ["hello", "hi", "hey", "good morning", "good evening"],
    "goodbye": ["bye", "see you later", "goodbye", "talk to you later"],
    "thanks": ["thanks", "thank you", "that's helpful", "thanks a lot"],
    "age": ["how old are you?", "what is your age?"],
    "name": ["what's your name?", "who are you?"],
    "weather": ["what's the weather?", "is it raining today?", "weather forecast"]
}

# Prepare data
X, y = [], []
for intent, patterns in intents.items():
    for pattern in patterns:
        X.append(pattern)
        y.append(intent)

# Shuffle the dataset
combined = list(zip(X, y))
random.shuffle(combined)
X[:], y[:] = zip(*combined)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Build pipeline: TF-IDF + Logistic Regression
model = Pipeline([
    ('tfidf', TfidfVectorizer()),
    ('clf', LogisticRegression(max_iter=200))
])

# Train
model.fit(X_train, y_train)

# Evaluate
preds = model.predict(X_test)
print("\nClassification Report:\n")
print(classification_report(y_test, preds))

# Save model
joblib.dump(model, "intent_classifier.joblib")
print("\n[+] Intent classifier saved as 'intent_classifier.joblib'")

# Sample usage
def predict_intent(text):
    model = joblib.load("intent_classifier.joblib")
    return model.predict([text])[0]

# Example prediction
if __name__ == '__main__':
    while True:
        query = input("You: ")
        if query.lower() in ["quit", "exit"]:
            break
        intent = predict_intent(query)
        print(f"Predicted Intent: {intent}\n")
