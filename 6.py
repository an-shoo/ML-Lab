import pandas as pd
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.model_selection import train_test_split

# Load the dataset
data = pd.read_csv('playtennis.csv')

# Prepare the features and labels
X = data.iloc[:, :-1]  # All columns except the last one
y = data['PlayTennis']  # Target column

# Convert categorical features into numerical values
X_encoded = pd.get_dummies(X)

# Train a decision tree classifier
X_train, X_test, y_train, y_test = train_test_split(X_encoded, y, test_size=0.2, random_state=42)
model = DecisionTreeClassifier(criterion="entropy", random_state=42)
model.fit(X_train, y_train)

# Print the decision tree
tree_rules = export_text(model, feature_names=list(X_encoded.columns))
print("Decision Tree Rules:\n", tree_rules)

# Classify a new sample
new_sample = X_encoded.iloc[0:1]  # Replace with actual sample
prediction = model.predict(new_sample)
print("Prediction for new sample:", prediction[0])
