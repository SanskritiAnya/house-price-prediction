import pandas as pd
from sklearn.model_selection import train_test_split

df = pd.read_csv("Housing.csv")

# one-hot encode categorical variables
df_encoded = pd.get_dummies(df, drop_first=True)

# separating features and target
X = df_encoded.drop("price", axis=1)
y = df_encoded["price"]

# spliting into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# confirm shapes
print("Training features shape:", X_train.shape)
print("Testing features shape:", X_test.shape)
print("Training labels shape:", y_train.shape)
print("Testing labels shape:", y_test.shape)
