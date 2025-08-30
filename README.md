# Implementation-of-Linear-Regression-Using-Gradient-Descent

## AIM:
To write a program to predict the profit of a city using the linear regression model with gradient descent.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Import the required library and read the dataframe.
2.Write a function computeCost to generate the cost function. 
3.Perform iterations og gradient steps with learning rate. 
4.Plot the Cost function using Gradient Descent and generate the required graph. 

## Program:
```
/*
Program to implement the linear regression using gradient descent.
Developed by: Shivani M
RegisterNumber: 212224040313
*/
```
```
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

# Function to perform Linear Regression using Gradient Descent
def linear_regression(X1, y, learning_rate=0.1, num_iters=1000):
    # Add a column of ones for the intercept term
    X = np.c_[np.ones(len(X1)), X1]
    
    # Initialize theta (model parameters) with zeros
    theta = np.zeros(X.shape[1]).reshape(-1, 1)
    
    # Gradient Descent loop
    for _ in range(num_iters):
        # Compute predictions (hypothesis)
        predictions = (X).dot(theta).reshape(-1, 1)
        
        # Compute errors (difference between predicted and actual values)
        errors = (predictions - y).reshape(-1, 1)
        
        # Update theta using gradient descent rule
        theta -= learning_rate * (1 / len(X1)) * X.T.dot(errors)
    
    return theta

# Load dataset
data = pd.read_csv("50_Startups.csv")
data.head()   # Just shows first few rows (not stored)

# Select features (all rows except first, excluding last 2 columns)
X = (data.iloc[1:, :-2].values)

# Convert features to float
X1 = X.astype(float)

# Create scaler object for standardization
scaler = StandardScaler()

# Target variable (last column of dataset, excluding first row)
y = (data.iloc[1:, -1].values).reshape(-1, 1)

# Standardize features
X1_Scaled = scaler.fit_transform(X1)

# Standardize target
Y1_Scaled = scaler.fit_transform(y)

# Print original features and scaled features
print(X)
print(X1_Scaled)

# Train model to learn parameters theta
theta = linear_regression(X1_Scaled, Y1_Scaled)

# New data point for prediction (3 feature values)
new_data = np.array([165349.2, 136897.8, 471784.1]).reshape(-1, 1)

# Scale new data (here you used fit_transform instead of transform)
new_Scaled = scaler.fit_transform(new_data)

# Add intercept term (1) and compute prediction
prediction = np.dot(np.append(1, new_Scaled), theta)

# Reshape prediction into column vector
prediction = prediction.reshape(-1, 1)

# Inverse scaling to get prediction back to original scale
pre = scaler.inverse_transform(prediction)

# Print raw prediction (scaled) and final predicted value
print(prediction)
print(f"Predicted value: {pre}")

```


## Output:
### Value of X
<img width="285" height="830" alt="image" src="https://github.com/user-attachments/assets/0cd8834a-0961-4f4a-89c2-27991c582875" />

### Value of Y
<img width="479" height="851" alt="image" src="https://github.com/user-attachments/assets/b1c3bb13-e5cd-4881-ab87-5fcc5cd84725" />

### Predicted Output 

<img width="324" height="22" alt="image" src="https://github.com/user-attachments/assets/bb2979f5-d569-4ec2-98f8-4234cf9e69bb" />

## Result:
Thus the program to implement the linear regression using gradient descent is written and verified using python programming.
