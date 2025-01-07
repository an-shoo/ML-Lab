import numpy as np
import matplotlib.pyplot as plt

# Define dataset
X = np.linspace(-3, 3, 100)
y = np.sin(X) + np.random.normal(0, 0.1, len(X))

# Locally Weighted Regression
def kernel(x, xi, tau):
    return np.exp(-np.sum((x - xi) ** 2) / (2 * tau ** 2))

def predict(X_train, y_train, x0, tau):
    weights = np.array([kernel(x0, xi, tau) for xi in X_train])
    W = np.diag(weights)
    theta = np.linalg.inv(X_train.T @ W @ X_train) @ X_train.T @ W @ y_train
    return x0 @ theta

# Fit the model
tau = 0.5
X_poly = np.vstack([np.ones_like(X), X]).T
y_pred = np.array([predict(X_poly, y, np.array([1, x]), tau) for x in X])

# Plot
plt.scatter(X, y, label="Data")
plt.plot(X, y_pred, color="red", label="LWR")
plt.legend()
plt.show()
