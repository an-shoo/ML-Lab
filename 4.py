import pandas as pd

# Load dataset
data = pd.read_csv("enjoysport.csv")
X = data.iloc[:, :-1].values
y = data.iloc[:, -1].values

# Find-S Algorithm
def find_s(X, y):
    # Initialize specific hypothesis to the first positive example
    specific_h = None
    for i in range(len(y)):
        if y[i] == "Yes":  # Check for positive example
            if specific_h is None:
                specific_h = X[i].copy()  # Set the first positive example as the hypothesis
            else:
                # Generalize the hypothesis for mismatched attributes
                specific_h = [
                    h if h == x else '?' for h, x in zip(specific_h, X[i])
                ]
    return specific_h

# Apply Find-S
print("Most Specific Hypothesis:", find_s(X, y))
