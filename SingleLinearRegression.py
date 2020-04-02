#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jan 19 23:32:50 2019

@author: fahadtariq
"""

#importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#importing the dataSet

dataset = pd.read_csv('Salary_Data.csv')

X = dataset.iloc[:, :-1].values
Y = dataset.iloc[:, 1].values


#Splitting Training and Test Set

from sklearn.model_selection import train_test_split

X_train,X_Test,Y_Train,Y_Test = train_test_split(X,Y,test_size = 1/3,random_state = 0 )

from sklearn.linear_model import LinearRegression
regressR = LinearRegression()
regressR.fit(X_train,Y_Train)

y_pred = regressR.predict(X_Test)

plt.scatter(X_train,Y_Train, color = 'red')
plt.plot(X_train ,regressR.predict(X_train), color = 'blue' )
plt.title('Salary vs Experiences (Training Set')
plt.xlabel('Years of Experience')
plt.ylabel('Salaries')
plt.show

plt.scatter(X_Test,Y_Test, color = 'red')
plt.plot(X_train ,regressR.predict(X_train), color = 'blue' )
plt.title('Salary vs Experiences (Training Set')
plt.xlabel('Years of Experience')
plt.ylabel('Salaries')
plt.show

#feature Scaling

"""from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train= sc_X.fit_transform(X_train)
X_Test = sc_X.transform(X_Test)""""