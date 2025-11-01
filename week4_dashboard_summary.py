# Week 4 - Dashboard Summary
# --------------------------

import pandas as pd
import matplotlib.pyplot as plt

# Load your task assignment and workload data
assignments = pd.read_csv('task_assignments.csv')

print("✅ Task Assignments Loaded\n")
print(assignments.head())

# Priority distribution
plt.figure(figsize=(6,4))
assignments['Priority'].value_counts().plot(kind='bar', color=['red','orange','green'])
plt.title('Task Distribution by Priority')
plt.xlabel('Priority')
plt.ylabel('Count')
plt.tight_layout()
plt.savefig('priority_distribution_summary.png')
plt.close()

# Tasks per user
plt.figure(figsize=(6,4))
assignments['Assigned_To'].value_counts().plot(kind='bar', color=['blue','purple','cyan'])
plt.title('Tasks per User')
plt.xlabel('User')
plt.ylabel('Number of Tasks')
plt.tight_layout()
plt.savefig('tasks_per_user_summary.png')
plt.close()

print("📊 Dashboard charts saved as:")
print(" - priority_distribution_summary.png")
print(" - tasks_per_user_summary.png")
print("\n✅ Week 4 Summary Dashboard Completed!")
