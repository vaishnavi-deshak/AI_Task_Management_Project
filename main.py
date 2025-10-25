# main.py
# Generate a synthetic task dataset and save to synthetic_tasks.csv

import pandas as pd
import random
import datetime

# 1. Task pool and categories
task_titles = [
    "Fix login bug", "Design dashboard UI", "Update API endpoint",
    "Write documentation", "Optimize database queries", "Test payment flow",
    "Deploy microservice", "Research new feature", "Refactor legacy code",
    "Set up CI/CD pipeline"
]

priorities = ["Low", "Medium", "High"]
statuses = ["Pending", "In Progress", "Completed"]
assignees = ["Alice", "Bob", "Charlie", "David", "Emma"]

# 2. Generate rows
data = []
for i in range(200):
    title = random.choice(task_titles)
    priority = random.choices(priorities, weights=[0.4, 0.4, 0.2])[0]
    status = random.choice(statuses)
    assignee = random.choice(assignees)
    days_due = random.randint(1, 30)
    created = datetime.date.today() - datetime.timedelta(days=random.randint(1, 60))
    deadline = created + datetime.timedelta(days=days_due)
    description = f"{title} - implement and test. Notes: {random.choice(['refactor', 'bugfix', 'enhancement', 'urgent', 'low risk'])}"
    data.append([i+1, description, priority, status, assignee, created.isoformat(), deadline.isoformat()])

# 3. Create DataFrame
df = pd.DataFrame(data, columns=[
    "Task_ID", "Task_Description", "Priority", "Status", "Assignee", "Created_Date", "Deadline"
])

# 4. Basic info and save
print("Dataset shape:", df.shape)
print(df.head().to_string(index=False))

df.to_csv("synthetic_tasks.csv", index=False)
print("✅ Dataset saved as synthetic_tasks.csv in this folder.")
