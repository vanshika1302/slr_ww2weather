#Weather during World War II

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#Import the Dataset
dataset = pd.read_csv("Summary of Weather.csv")
X = dataset.iloc[:, 5].values
y = dataset.iloc[:, 4].values

X = X.reshape(-1, 1)
y = y.reshape(-1, 1)

#Splitting into training set and test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)

from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)

y_pred = regressor.predict(X_test)

regressor.score(X, y)

plt.scatter(X_test, y_test, color='black')
plt.plot(X_test, regressor.predict(X_test), color='blue')
plt.title('MaxTemp vs. MinTemp (Training Set)')
plt.xlabel('MinTemp')
plt.ylabel('MaxTemp')
plt.show()