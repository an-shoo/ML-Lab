from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
import pandas as pd

# 1. Import dataset
iris = load_iris()
data = pd.DataFrame(iris.data, columns=iris.feature_names)
data['Species'] = iris.target

# 2. Display first 5 rows
print(data.head())

# 3. Check number of samples for each class
print(data['Species'].value_counts())

# 4. Check for null values
print(data.isnull().sum())

# 5. Visualize data
data['Species'].value_counts().plot(kind='bar')

# 6. Covariance and correlation
print("Covariance:\n", data.cov())
print("Correlation:\n", data.corr())

# 7. Train-test split
X = data.drop('Species', axis=1)
y = data['Species']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 8. Train model
model = LogisticRegression()
model.fit(X_train, y_train)

# 9. Predict and evaluate
y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred))
