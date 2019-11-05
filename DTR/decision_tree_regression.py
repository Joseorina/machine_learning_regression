#importing the libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#Importing teh dataset
dataset = pd.read_csv('Position_Salaries.csv')
X = dataset.iloc[:, 1:2 ].values
y = dataset.iloc[:,2].values

#Fitting the DTR to the dataset
from sklearn.tree import DecisionTreeRegressor
regressor = DecisionTreeRegressor(random_state=0)
regressor.fit(X, y)

#Predicting a new result
y_pred = regressor.predict([[6.5]])

#Visualizing the DTR results
plt.scatter(X, y, color='red')
plt.plot(X, regressor.predict(X), color = 'blue')
plt.title('Truth or Bluff(Decision Trees)')
plt.xlabel('Position label')
plt.ylabel('Salary')
plt.show()

#Higher resolution plot
X_grid = np.arange(min(X), max(X), 0.01)
X_grid = X_grid.reshape(len(X_grid), 1)
plt.scatter(X, y, color='red')
plt.plot(X_grid, regressor.predict(X_grid), color='blue')
plt.title('Decision Tree REgressio')
plt.xlabel('Position label')
plt.ylabel('Salary')
plt.show()