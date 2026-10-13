import pandas as pd

# Load dataset
data = pd.read_csv("data/leads.csv")

# Display first 5 rows
print("First 5 rows:")
print(data.head())

# Display number of rows and columns
print("\nDataset shape:")
print(data.shape)

# Display column names
print("\nColumn names:")
print(data.columns)

# Display information about the dataset
print("\nDataset information:")
print(data.info())