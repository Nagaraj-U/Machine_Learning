# -*- coding: utf-8 -*-
"""
Created on Sun Mar 15 14:36:05 2020

@author: Nagaraj U
"""
#importing libraries

import numpy as np  #for math calculations
import matplotlib.pyplot as plt
import pandas as pd  #for datasets

#importing dataset

dataset= pd.read_csv('Data.csv')   
X=dataset.iloc[:, :-1].values
y=dataset.iloc[:, 3].values       

#missing data handling
from sklearn.preprocessing import Imputer     #import Imputer class from sklearn.preprocessing library
imputer=Imputer(missing_values='NaN' ,strategy='mean' ,axis=0)  #press ctrl+i to know about parameter info
imputer=imputer.fit(X[:, 1:3])  #only second and third column has to be filled  upper bound(3) is neglected in python and first colon for all the values of rowss
X[:, 1:3]=imputer.transform(X[:, 1:3])  #for tranforming column and store back to original array

#catagorising data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder  #to catagorise data in first and third column character to number to help for calculations
labelencoder_X=LabelEncoder()
X[:, 0]=labelencoder_X.fit_transform(X[:, 0])
onehotencoder=OneHotEncoder(categorical_features=[0])
X= onehotencoder.fit_transform(X).toarray()   #storing back to X array 

labelencoder_y=LabelEncoder()
y=labelencoder_y.fit_transform(y)  #1 for yes and 0 for no

#splitting dataset into training set and test set
#training set is generally larger by which ML model learns and test set is the one by which machine learning model tets

from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X, y, test_size=0.2, random_state=0)

#feature scaling  (helps to get all the values in the same range normally 0-1 that increse efficiency of algo)
from sklearn.preprocessing import StandardScaler
Sc_X=StandardScaler()
X_train=Sc_X.fit_transform(X_train)
X_test=Sc_X.transform(X_test)
