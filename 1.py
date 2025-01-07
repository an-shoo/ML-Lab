# Importing libraries
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Step 1: Import dataset
data = fetch_california_housing(as_frame=True)
df = data.frame

# Step 2: Display first 5 rows
print("First 5 rows of the dataset:")
print(df.head())

# Step 3: Check the number of samples (not relevant for regression, skipped)

# Step 4: Check for null values
print("\nNull values in the dataset:")
print(df.isnull().sum())

# Step 5: Visualize the data in the form of graphs
sns.pairplot(df.sample(500))  # Use a sample for visualization
plt.show()

# Visualizing distribution of the target variable
sns.histplot(df['MedHouseVal'], kde=True, bins=30)
plt.title("Distribution of Target Variable (MedHouseVal)")
plt.show()

# Step 6: Obtain covariance and correlation values
cov_matrix = df.cov()
correlation_matrix = df.corr()

print("\nCovariance Matrix (partial):")
print(cov_matrix.head())
print("\nCorrelation Matrix (partial):")
print(correlation_matrix.head())

# Heatmap of the correlation matrix
sns.heatmap(correlation_matrix, annot=True, fmt='.2f', cmap='coolwarm')
plt.title("Correlation Matrix Heatmap")
plt.show()

# Step 7: Train and test model
X = df.drop(columns='MedHouseVal')  # Features
y = df['MedHouseVal']  # Target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 8: Apply regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Step 9: Predict and evaluate accuracy
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"\nMean Squared Error: {mse:.2f}")
print(f"R² Score: {r2:.2f}")
