🤖 AI Task Management System

This project uses Natural Language Processing (NLP) and Machine Learning to automate task management.
It analyzes task descriptions, predicts their priority, and distributes them among users for balanced workload management.

🚀 Features

✅ Text preprocessing and NLP tokenization
✅ Task priority prediction using Random Forest
✅ Task classification using Naive Bayes
✅ Workload balancing among users
✅ Visual dashboards for task insights

🧠 Tech Stack
Python 3.12+
Pandas, NumPy – data processing
NLTK – text cleaning and tokenization
Scikit-learn – model building (Naive Bayes, Random Forest)
Matplotlib, Seaborn – data visualization
Git & GitHub – version control

⚙️ How It Works

1.Data Preprocessing – cleaned and tokenized text using NLP.
2.Feature Extraction – converted text to numerical form using TF-IDF.
3.Model Training – trained Naive Bayes and Random Forest models.
4.Prediction – predicted task priority (High / Medium / Low).
5.Visualization – generated charts:
-priority_distribution_summary.png
-tasks_per_user_summary.png

📊 Results
Models achieved up to 100% accuracy on training data.
Dashboard visuals confirm accurate workload distribution.

📂 Project Structure
AI_TASK_MANAGEMENT/
│
├── main.py
├── eda_cleaning.py
├── day5_7.py
├── week2_model.py
├── processed_text.csv
├── priority_distribution_summary.png
├── tasks_per_user_summary.png
└── README.md

📸 Screenshots


<img width="600" height="400" alt="priority_distribution" src="https://github.com/user-attachments/assets/b5bc055d-7fc5-4387-971e-b93ede05dd56" />


<img width="600" height="400" alt="priority_distribution_summary" src="https://github.com/user-attachments/assets/e52ac542-4217-4778-a5b6-671bd40c8c9e" />


<img width="700" height="500" alt="status_by_priority" src="https://github.com/user-attachments/assets/63dd48f1-2dcd-400b-bd46-f6cc17f2503d" />


<img width="600" height="400" alt="tasks_per_user_summary" src="https://github.com/user-attachments/assets/53601c0f-d33a-4aae-bf68-a4ac81d44225" />


💬 Future Scope

🚧 Deploy model using Streamlit or Flask for real-time task management.
📈 Expand dataset and implement deep learning for advanced prediction.

👩‍💻 Author

Vaishnavi Deshak
Data Science Internship Project – AI Task Management System







