# Implementation-of-Linear-Regression-Using-Gradient-Descent

## AIM:
To write a program to predict the profit of a city using the linear regression model with gradient descent.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Initialize parameters m (slope) and b (intercept) to 0.
2. Normalize input data and compute predicted values ypred = m * X + b.
3. Update m and b using gradient descent to minimize error.
4. Repeat for given epochs and obtain final optimized m and b.

## Program:
```
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

data = pd.read_csv("Startup.csv")

X = data['R&D Spend'].values
y = data['Profit'].values

X = (X - X.mean()) / X.std()

m = 0
b = 0

learning_rate = 0.01
epochs = 1000
n = len(X)

for i in range(epochs):
    y_pred = m * X + b

    dm = (-2/n) * np.sum(X * (y - y_pred))
    db = (-2/n) * np.sum(y - y_pred)

    m = m - learning_rate * dm
    b = b - learning_rate * db

print("Slope (m):", m)
print("Intercept (b):", b)

y_pred = m * X + b

plt.scatter(X, y)
plt.plot(X, y_pred)

plt.xlabel("R&D Spend (Normalized)")
plt.ylabel("Profit")
plt.title("Gradient Descent on 50_Startups Dataset")

plt.show()

```
Developed by: Mohammed Ameer F
RegisterNumber:  212225040248


## Output:
<img width="755" height="557" alt="Screenshot 2026-04-22 093124" src="https://github.com/user-attachments/assets/8e8233d8-8e8e-473c-b2a9-0bcb4860d055" />



## Result:
Thus the program to implement the linear regression using gradient descent is written and verified using python programming.
