# Week 2 - Task Classification using TF-IDF + Naive Bayes
# -------------------------------------------------------

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report

# Step 1: Load the processed dataset with labels
data = pd.read_csv('processed_text_with_labels.csv')

print("✅ Data loaded successfully!")
print("Columns:", data.columns)
print(data.head())

# Step 2: Use the cleaned text column as input features
X = data['cleaned_text']
y = data['priority']

# Step 3: Convert text to TF-IDF features
vectorizer = TfidfVectorizer(max_features=500)
X_tfidf = vectorizer.fit_transform(X)

# Step 4: Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X_tfidf, y, test_size=0.2, random_state=42, stratify=y
)


# Step 5: Train Naive Bayes model
model = MultinomialNB()
model.fit(X_train, y_train)

# Step 6: Make predictions
y_pred = model.predict(X_test)

# Step 7: Evaluate performance
print("\n📊 Model Evaluation Results:")
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nDetailed Classification Report:")
print(classification_report(y_test, y_pred))

print("✅ Week 2 Naive Bayes classification completed successfully!")
