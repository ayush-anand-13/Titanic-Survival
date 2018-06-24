#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 22 06:22:51 2018

@author: ayush
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('train.csv')
y = dataset.iloc[:,1].values
X = dataset.iloc[:,[2,4,5,6,7,9]].values

#Encoding data
from sklearn.preprocessing import LabelEncoder,OneHotEncoder
Gender = LabelEncoder()
X[:,1] = Gender.fit_transform(X[:,1])

#replace nan values here
from sklearn.preprocessing import Imputer
imputer = Imputer(missing_values = 'NaN', strategy = 'mean', axis = 0)
imputer = imputer.fit(X)
X = imputer.transform(X)




#feature scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X = sc.fit_transform(X)

from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score
from keras.models import Sequential
from keras.layers import Dense
def build_classifier():
    classifier = Sequential()
    classifier.add(Dense(units = 6, kernel_initializer = 'uniform', activation = 'relu', input_dim = 6))
    classifier.add(Dense(units = 6, kernel_initializer = 'uniform', activation = 'relu'))
    classifier.add(Dense(units = 6, kernel_initializer = 'uniform', activation = 'relu'))
    classifier.add(Dense(units = 6, kernel_initializer = 'uniform', activation = 'relu'))
    classifier.add(Dense(units = 1, kernel_initializer = 'uniform', activation = 'sigmoid'))
    classifier.compile(optimizer = 'rmsprop', loss = 'binary_crossentropy', metrics = ['accuracy'])
    return classifier
classifier = KerasClassifier(build_fn = build_classifier, batch_size = 25, epochs = 500)
accuracies = cross_val_score(estimator = classifier, X = X, y = y, cv = 10, n_jobs = -1)
mean = accuracies.mean()
variance = accuracies.std()

classifier1 = Sequential()

# Adding the input layer and the first hidden layer
classifier1.add(Dense(units = 4, kernel_initializer = 'uniform', activation = 'relu', input_dim = 6))
# classifier.add(Dropout(p = 0.1))

# Adding the second hidden layer
classifier1.add(Dense(units = 4, kernel_initializer = 'uniform', activation = 'relu'))
# classifier.add(Dropout(p = 0.1))

# Adding the output layer
classifier1.add(Dense(units = 1, kernel_initializer = 'uniform', activation = 'sigmoid'))

# Compiling the ANN
classifier1.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

# Fitting the ANN to the Training set
classifier1.fit(X, y, batch_size = 25, epochs = 500)
#INput test set
dataset2 =pd.read_csv('test.csv')
X_test = dataset2.iloc[:,[1,3,4,5,6,8]].values

#Find and label genders
Gender2 = LabelEncoder()
X_test[:,1] = Gender2.fit_transform(X_test[:,1])

#Input missing values
imputer = Imputer(missing_values = 'NaN', strategy = 'mean', axis = 0)
imputer = imputer.fit(X_test)
X_test = imputer.transform(X_test)

#Transform the data
X_test = sc.transform(X_test)

#Predictboi
y_pred = classifier1.predict(X_test)
y_pred = (y_pred > 0.5)

Survive = LabelEncoder()
ybool = Gender2.fit_transform(y_pred)

xboi = dataset2.iloc[:,0].values

ans =  np.concatenate((xboi,ybool ), axis=0)
ans = ans.reshape(2,418)
ans = ans.T

np.savetxt("foo4.csv", ans, delimiter=",")