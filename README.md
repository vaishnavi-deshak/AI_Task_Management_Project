# 🤖 **AI Task Management System**
AI Task Management System is a data science project developed as part of my Data Science Internship.
It uses Machine Learning and Natural Language Processing (NLP) to automatically predict the priority level of tasks — High, Medium, or Low — based on their textual descriptions.
The project helps teams manage workload efficiently by identifying which tasks require immediate attention and which can be scheduled later

# 🚀** Features**
Predicts task priority using trained ML models
Real-time prediction via Streamlit Web App
Performs text preprocessing (stopword removal, tokenization, TF-IDF vectorization)
Saves trained models (.pkl files) for reuse
Includes data visualization for task distribution

# 🧠** Tech Stack**
Languages: Python
Libraries: Pandas, NumPy, Scikit-learn, NLTK, Joblib, Streamlit
Tools: VS Code, GitHub
Model: Random Forest Classifier

# ⚙️** How It Works**
1.Data Cleaning & Preprocessing: Removes noise and prepares text data.
2.Feature Extraction: Converts text to numeric form using TF-IDF.
3.Model Training: Trains Random Forest model for priority classification.
4.Deployment: Streamlit app provides an interactive interface for users to enter tasks and instantly get priority predictions.

#💻** Run Locally**
git clone https://github.com/vaishnavi-deshak/AI_Task_Management_Project.git
cd AI_Task_Management_Project
pip install -r requirements.txt
python -m streamlit run app.py

#📊** Output**
-Enter a task like:
“Fix urgent issue in client system” → High Priority
“Review weekly report” → Medium Priority
“Organize old files” → Low Priority

# 📸** Screenshots**

<img width="600" height="400" alt="priority_distribution_summary" src="https://github.com/user-attachments/assets/e52ac542-4217-4778-a5b6-671bd40c8c9e" />

<img width="700" height="500" alt="status_by_priority" src="https://github.com/user-attachments/assets/63dd48f1-2dcd-400b-bd46-f6cc17f2503d" />

# 👩‍💻** Author**

Vaishnavi Deshak
Data Science Internship Project – AI Task Management System







