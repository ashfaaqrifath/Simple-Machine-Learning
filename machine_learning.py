import numpy as np
from sklearn.linear_model import LinearRegression

# Create a dataset
X = np.array([1, 2, 3, 4, 5]).reshape(-1, 1)
y = np.array([2, 4, 5, 4, 5])

# Create a linear regression model
model = LinearRegression()

# Fit the model to the dataset
model.fit(X, y)

# Make a prediction
x_test = np.array([[6]])
y_pred = model.predict(x_test)

# Print the predicted value
print(y_pred)