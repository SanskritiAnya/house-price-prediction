import pandas as pd

df = pd.read_csv("housing.zip") 
print("First 5 rows of the dataset:")
print(df.head())

#checking for missing values
print("\nMissing values:")
print(df.isnull().sum())

#dropping rows with missing values
df_cleaned = df.dropna()

# Save cleaned dataset
df_cleaned.to_csv("cleaned_data.csv", index=False)
print("\n Cleaned data saved to cleaned_data.csv")
