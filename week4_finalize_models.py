# Week 4 - Finalize and Save Models
# ---------------------------------

import pandas as pd
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# Load data
data = pd.read_csv('processed_text_with_labels.csv')
X = data['text']
y = data['priority']

# TF-IDF
vectorizer = TfidfVectorizer(max_features=500)
X_tfidf = vectorizer.fit_transform(X)

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X_tfidf, y, test_size=0.2, random_state=42, stratify=y
)

# Train Naive Bayes
nb_model = MultinomialNB()
nb_model.fit(X_train, y_train)

# Train Random Forest (using best params from Week 3)
rf_model = RandomForestClassifier(
    n_estimators=200, max_depth=10, min_samples_split=2,
    min_samples_leaf=1, random_state=42
)
rf_model.fit(X_train, y_train)

# Save both models + vectorizer
joblib.dump(nb_model, 'naive_bayes_model.pkl')
joblib.dump(rf_model, 'random_forest_model.pkl')
joblib.dump(vectorizer, 'tfidf_vectorizer.pkl')

print("✅ Models and vectorizer saved successfully!")
