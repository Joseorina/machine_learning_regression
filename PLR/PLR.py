#Polynomial Regression

#Data Preprocessing
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#Imorting the data set
dataset = pd.read_csv('Position_Salaries.csv')
X = dataset.iloc[:,1:2].values
y = dataset.iloc[:,2].values

#Splitting the dataset into Training set and Test set
"""
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=0)
"""

#Fitting Linear Regression to the dataset
from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
lin_reg.fit(X,y)

#Fitting Polynomial Regression to the dataset
#Transform into x_poly
from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree = 4)
X_poly = poly_reg.fit_transform(X)

#create new LR object and fit X_poly
lin_reg_2 = LinearRegression()
lin_reg_2.fit(X_poly, y)

#Visualising the Linear REgression REsults
plt.scatter(X, y, color='red')
plt.plot(X, lin_reg.predict(X), color = 'blue')
plt.title('Truth or Bluuf(Linear Regression)')
plt.xlabel('Position Level')
plt.ylabel('Salary')
plt.show()


#Visualizing the polynomial Regression Results
X_grid = np.arange(min(X),max(X), 0.1)
X_grid = X_grid.reshape((len(X_grid),1))
plt.scatter(X, y, color='red')
plt.plot(X_grid, lin_reg_2.predict(poly_reg.fit_transform(X_grid)), color='blue')
plt.title('Truth or Bluuf(Linear Regression)')
plt.xlabel('Position Level')
plt.ylabel('Salary')
plt.show()

#Predicting a new result with Linear Regression
lin_reg.predict(array.reshape(6.5).reshape(1,-1)

#Predicting a new value with Polynomial Regression
lin_reg_2.predict(poly_reg.fit_transform(6.5))
