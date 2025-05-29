import pandas as pd
from sklearn.linear_model import LinearRegression

df = pd.read_csv("Housing.csv")
df_encoded = pd.get_dummies(df, drop_first=True)
X = df_encoded.drop("price", axis=1)

model = LinearRegression()
model.fit(X, df_encoded["price"])

coefficients = pd.DataFrame({
    "Feature": X.columns,
    "Coefficient": model.coef_
}).sort_values(by="Coefficient", ascending=False)

print(coefficients)
