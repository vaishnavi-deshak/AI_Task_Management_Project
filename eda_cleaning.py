# eda_cleaning.py
# Exploratory Data Analysis (EDA) + Data Cleaning for AI Task Management dataset

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import nltk
from nltk.corpus import stopwords
import os, re

# If dataset exists, load it
if os.path.exists("synthetic_tasks.csv"):
    df = pd.read_csv("synthetic_tasks.csv")
else:
    raise FileNotFoundError("❌ synthetic_tasks.csv not found. Run main.py first.")

print("\n✅ Dataset loaded successfully!")
print("Shape:", df.shape)
print("\n--- First 5 rows ---")
print(df.head().to_string(index=False))

# ---------- Basic EDA ----------
print("\n--- Value counts ---")
print("\nPriority:\n", df['Priority'].value_counts())
print("\nStatus:\n", df['Status'].value_counts())
print("\nAssignee:\n", df['Assignee'].value_counts())

# Convert dates to datetime
df['Created_Date'] = pd.to_datetime(df['Created_Date'])
df['Deadline'] = pd.to_datetime(df['Deadline'])

# Compute days until deadline
df['Days_to_Deadline'] = (df['Deadline'] - pd.Timestamp.today()).dt.days

# ---------- Plot visualizations ----------
sns.set(style="whitegrid")

# Priority distribution
plt.figure(figsize=(6,4))
sns.countplot(data=df, x='Priority', order=['Low','Medium','High'])
plt.title("Task Priority Distribution")
plt.tight_layout()
plt.savefig("priority_distribution.png")
plt.close()
print("📊 Saved: priority_distribution.png")

# Status by Priority
plt.figure(figsize=(7,5))
sns.countplot(data=df, x='Priority', hue='Status')
plt.title("Status by Priority")
plt.tight_layout()
plt.savefig("status_by_priority.png")
plt.close()
print("📊 Saved: status_by_priority.png")

# ---------- Text Cleaning ----------
nltk.download('stopwords', quiet=True)
stops = set(stopwords.words('english'))

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'[^a-z\s]', ' ', text)  # keep only letters
    tokens = [t for t in text.split() if t not in stops and len(t) > 1]
    return tokens

df['Tokens'] = df['Task_Description'].apply(clean_text)

# Check most common words
all_tokens = [t for tokens in df['Tokens'] for t in tokens]
most_common = Counter(all_tokens).most_common(10)
print("\n🔠 Most common tokens:", most_common)

# ---------- Save cleaned dataset ----------
df.to_csv("synthetic_tasks_cleaned.csv", index=False)
print("\n✅ Cleaned dataset saved as synthetic_tasks_cleaned.csv")
