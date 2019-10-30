#Simple Linear Regression

#Importting the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#Importing datasets
dataset = pd.read_csv('Salary_Data.csv')
#Selecting idepende variables into a feature matrix
X = dataset.iloc[:, :-1].values
#choosing the dependet vatriable vector
y = dataset.iloc[:,1].values

#splitting dataset into Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=1/3,random_state=0)

#Fitting SLR to the traiing set
from sklearn.linear_model import LinearRegression
#The machine
regressor = LinearRegression()
#The machine regressor learning based on test data
regressor.fit(X_train, y_train)

#Predicting the Test set result
y_pred = regressor.predict(X_test)

#Visualising the Training set result
plt.scatter(X_train, y_train, color='red')
plt.plot(X_train, regressor.predict(X_train), color='blue')
plt.title('Salary vs Experience(Training set)')
plt.xlabel('Years of experience')
plt.ylabel('Salary')
plt.show()

#Visualising the Test set result
plt.scatter(X_test, y_test, color='red')
plt.plot(X_train, regressor.predict(X_train), color='blue')
plt.title('Salary vs Experience(Test Set)')
plt.xlabel('Years of Exprerience')
plt.ylabel('Salary')
plt.show()