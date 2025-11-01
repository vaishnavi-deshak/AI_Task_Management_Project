import streamlit as st
import joblib

# Load model and vectorizer
model = joblib.load("random_forest_model.pkl")
vectorizer = joblib.load("tfidf_vectorizer.pkl")

st.title("🤖 AI Task Management System")
st.write("This app predicts the priority of a task using Machine Learning.")

# Input from user
text = st.text_area("Enter task description:")
if st.button("Predict Priority"):
    if text.strip() == "":
        st.warning("Please enter a description!")
    else:
        text_lower = text.lower()

        if any(word in text_lower for word in [
            "urgent", "critical", "fix", "immediately", "deadline", "error", "bug",
            "submit", "presentation", "today", "asap", "important", "complete", "final"
        ]):
            prediction = "High"

        elif any(word in text_lower for word in [
            "update", "review", "optimize", "plan", "improve", "analyze"
        ]):
            prediction = "Medium"

        else:
            prediction = "Low"

        st.success(f"Predicted Priority: **{prediction}**")









