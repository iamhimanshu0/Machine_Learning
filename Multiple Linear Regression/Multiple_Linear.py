# -*- coding: utf-8 -*-
"""
Created on Tue Jan 29 10:52:13 2019

@author: himanshu
"""
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset = pd.read_csv("50_Startups.csv")
X = dataset.iloc[:,:-1].values
y = dataset.iloc[:,4].values

#Encoding the categorical value
from sklearn.preprocessing import LabelEncoder , OneHotEncoder
labelencoder_X = LabelEncoder()
X[:,3] = labelencoder_X.fit_transform(X[:,3])
onehotencoder = OneHotEncoder(categorical_features=[3])
X = onehotencoder.fit_transform(X).toarray()

#Avoiding the Dummy Variable Trap
X = X[:,1:]

#Splitting the dataset into Trainning and Testing
from sklearn.model_selection import train_test_split
X_train, X_test , y_train, y_test = train_test_split(X,y,test_size = 0.2,random_state = 0)

from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train,y_train)

#predicting the test values
y_pred = regressor.predict(X_test)