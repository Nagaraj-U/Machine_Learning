# -*- coding: utf-8 -*-
"""
Created on Tue Mar 24 23:57:28 2020

@author: Nagaraj U
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset=pd.read_csv('Position_Salaries.csv')
X=dataset.iloc[:,1:2].values
y=dataset.iloc[:,2].values

#no need of splitting into train and test

#fitting random forest tree for dataset
from sklearn.ensemble import RandomForestRegressor
regressor=RandomForestRegressor(n_estimators=500,random_state=0)
regressor.fit(X,y)

#preficting result
y_pred=regressor.predict([[6.5]])

#visualising results
X_grid=np.arange(min(X),max(X),0.01)
X_grid=X_grid.reshape(len(X_grid),1)
plt.scatter(X,y,color='red')
plt.plot(X_grid,regressor.predict(X_grid),color='blue')
plt.title('truth or bluf of emplyee')
plt.x_label('level')
plt.y_label('salaries')