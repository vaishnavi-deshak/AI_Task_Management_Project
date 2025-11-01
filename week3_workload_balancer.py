# Week 3 - Workload Balancing Logic
# ---------------------------------

import pandas as pd
import random

# Step 1: Load the dataset with priority predictions
data = pd.read_csv('processed_text_with_labels.csv')

# Simulate a few users in the system
users = ['Alice', 'Bob', 'Charlie']

# Step 2: Create a workload tracker
# Each user starts with zero tasks and total workload 0
workload = {user: {'tasks': 0, 'score': 0} for user in users}

# Step 3: Define a numeric weight for each priority
priority_weights = {'High': 3, 'Medium': 2, 'Low': 1}

assignments = []

# Step 4: Assign tasks based on lowest workload score
for idx, row in data.iterrows():
    # pick user with least total workload
    chosen_user = min(workload, key=lambda u: workload[u]['score'])

    # calculate weight
    weight = priority_weights.get(row['priority'], 1)

    # update workload
    workload[chosen_user]['tasks'] += 1
    workload[chosen_user]['score'] += weight

    # store assignment
    assignments.append({'Task': row['text'], 'Priority': row['priority'], 'Assigned_To': chosen_user})

# Step 5: Save the new assignments
assign_df = pd.DataFrame(assignments)
assign_df.to_csv('task_assignments.csv', index=False)

# Step 6: Show summary
print("✅ Workload balancing completed!\n")
print(assign_df.head(10))
print("\n📊 User workload summary:")
for user, info in workload.items():
    print(f"{user}: {info['tasks']} tasks, total workload {info['score']}")
