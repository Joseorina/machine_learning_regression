import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#Importing teh dataset
dataset = pd.read_csv('Position_Salaries.csv')
X = dataset.iloc[:, 1:2].values
y = dataset.iloc[:,2].values

#Feature scalling
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
sc_y = StandardScaler()
X = sc_X.fit_transform(X)
y = sc_y.fit_transform(y)

#Fitting the SVR into the dataset
#Mackes model non linear
from sklearn.svm import SVR
regressor = SVR(kernel='rbf')
regressor.fit(X, y)

#Predicting a new result
y_pred = regressor.predict([[6.5]])

#Fitting SVR to the dataset

#Visualising a new result
plt.scatter(X, y, color='red')
plt.plot(X, regressor.predict(X), color='blue')
plt.title('Truth pr Bluff')
plt.xlabel('Position label')
plt.ylabel('Salary')
plt.show()