# -*- coding: utf-8 -*-
"""
Created on Wed Mar 18 14:53:57 2020

@author: Nagaraj U
"""

# Data Preprocessing Template

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Position_Salaries.csv')
X = dataset.iloc[:, 1:2].values
y = dataset.iloc[:, 2].values

#fitting linear regression to the dataset
from sklearn.linear_model import LinearRegression
lin_reg=LinearRegression()
lin_reg.fit(X,y)

#fitting polynomial regression to the dataset
from sklearn.preprocessing import PolynomialFeatures
poly_reg=PolynomialFeatures(degree=4)
X_poly=poly_reg.fit_transform(X)
lin_reg_2=LinearRegression()
lin_reg_2.fit(X_poly,y)

#visualising the linear regression results
plt.scatter(X,y,color='red')
plt.plot(X,lin_reg.predict(X),color='blue')  #here  lin_reg.predict(X)=X_poly 
plt.title('truth or bluf(salaries)')
plt.xlabel('level')
plt.ylabel('salries')
plt.show()

#visualising polynomial regression results
plt.scatter(X,y,color='red')
plt.plot(X,lin_reg_2.predict(poly_reg.fit_transform(X)),color='blue')
plt.title('truth or bluf(salaries)')
plt.xlabel('level')
plt.ylabel('salries')
plt.show()

#predicting new result with linear regression
lin_reg.predict(X)   #return 10 salaries for the diff levels acc to linear regression model

#predicing new result(salary) with polynomial regression
lin_reg_2.predict(poly_reg.fit_transform(X))  #return 10 salaries predicted by polynomial regression model


# Splitting the dataset into the Training set and Test set
'''
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)
'''
# Feature Scaling
"""from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)
sc_y = StandardScaler()
y_train = sc_y.fit_transform(y_train)"""