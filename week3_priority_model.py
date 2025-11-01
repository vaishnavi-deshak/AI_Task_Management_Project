# Week 3 - Priority Prediction Model using Random Forest
# -------------------------------------------------------

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score

# Step 1: Load the processed dataset with labels
data = pd.read_csv('processed_text_with_labels.csv')

print("✅ Data loaded successfully!")
print("Columns:", data.columns)
print(data.head())

# Step 2: Use text data and priority as target
X = data['text']              # features
y = data['priority']          # target

# Step 3: Convert text to TF-IDF features
vectorizer = TfidfVectorizer(max_features=500)
X_tfidf = vectorizer.fit_transform(X)

# Step 4: Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X_tfidf, y, test_size=0.2, random_state=42, stratify=y
)

# Step 5: Build and train Random Forest model
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

# Step 6: Make predictions
y_pred = rf_model.predict(X_test)

# Step 7: Evaluate model performance
print("\n📊 Model Evaluation Results:")
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nDetailed Classification Report:")
print(classification_report(y_test, y_pred))

print("✅ Week 3 Random Forest model training completed successfully!")

import joblib

# Save model and vectorizer
joblib.dump(rf_model, "random_forest_model.pkl")
joblib.dump(vectorizer, "tfidf_vectorizer.pkl")

print("✅ Model and vectorizer saved successfully!")
