from sklearn.datasets import fetch_california_housing
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

# Load dataset
housing = fetch_california_housing()
X, y = housing.data, housing.target
columns = housing.feature_names

# Convert to DataFrame
df = pd.DataFrame(X, columns=columns)

# 1. Display first 5 rows
print(df.head())

# 2. Check for null values
print("Null values:", df.isnull().sum())

# 3. Visualize data
df.hist(figsize=(10, 8))
plt.show()

# 4. Covariance and Correlation
print("Covariance:\n", df.cov())
print("Correlation:\n", df.corr())

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Random Forest
rf = RandomForestRegressor()
rf.fit(X_train, y_train)

# Predict and evaluate
y_pred = rf.predict(X_test)
print("Mean Squared Error:", mean_squared_error(y_test, y_pred))
