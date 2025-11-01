# Week 3 - Hyperparameter Tuning for Random Forest
# -------------------------------------------------

import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# Step 1: Load dataset
data = pd.read_csv('processed_text_with_labels.csv')
print("✅ Data loaded successfully!")

# Step 2: Prepare features and labels
X = data['text']
y = data['priority']

# Convert text into TF-IDF features
vectorizer = TfidfVectorizer(max_features=500)
X_tfidf = vectorizer.fit_transform(X)

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X_tfidf, y, test_size=0.2, random_state=42, stratify=y
)

# Step 3: Define base model and parameter grid
rf = RandomForestClassifier(random_state=42)

param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [None, 5, 10],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

# Step 4: Apply GridSearchCV
grid_search = GridSearchCV(
    estimator=rf,
    param_grid=param_grid,
    cv=3,
    n_jobs=-1,
    verbose=2
)

print("\n🔍 Running GridSearchCV... please wait, this may take a minute.")
grid_search.fit(X_train, y_train)

# Step 5: Evaluate best model
best_model = grid_search.best_estimator_
y_pred = best_model.predict(X_test)

print("\n✅ Grid Search Completed!")
print("Best Parameters:", grid_search.best_params_)
print("\n📊 Model Evaluation:")
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nDetailed Report:")
print(classification_report(y_test, y_pred))

print("\n✅ Week 3 - Step 3 (Hyperparameter Tuning) Completed Successfully!")
