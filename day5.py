import pandas as pd  # pandas helps to work with CSV files

# Load your CSV file
data = pd.read_csv('test.csv')  # make sure the filename matches your CSV

# Check the first few rows
print("Here is your data:")
print(data.head())
