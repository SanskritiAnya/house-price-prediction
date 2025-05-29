import pandas as pd

df = pd.read_csv("Housing.csv")

# data preview
print("Shape of the dataset:", df.shape)
print("\nFirst 5 rows:")
print(df.head())

# Check for missing values
print("\nMissing values per column:")
print(df.isnull().sum())




