#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 10 21:12:37 2018

@author: shibinmak
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


data = pd.read_csv('Churn_Modelling.csv')
#x = data.drop(['RowNumber','CustomerId','Surname','Exited'],axis=1).values
x= data.iloc[:,3:13]
y= data['Exited'].values

'''from sklearn.preprocessing import LabelEncoder,OneHotEncoder

labelencoder_region = LabelEncoder()
x.iloc[:,1] =labelencoder_region.fit_transform(x.iloc[:,1])
labelencoder_gender = LabelEncoder()
x.iloc[:,2] =labelencoder_gender.fit_transform(x.iloc[:,2])
onehotencoder= OneHotEncoder(categorical_features=[1])
x =onehotencoder.fit_transform(x).toarray()'''

geography = pd.get_dummies(x.iloc[:,1],drop_first=True)
gender = pd.get_dummies(x.iloc[:,2],drop_first=True)
x.drop(['Geography','Gender'],axis=1,inplace=True)
x= pd.concat([x,geography,gender],axis=1)
xx=x
x=x.values

from sklearn.preprocessing import StandardScaler
scaler=StandardScaler()
x= scaler.fit_transform(x)

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.3,random_state=33)

import keras
from keras.models import Sequential
from keras.layers import Dense

classifier =Sequential()

classifier.add(Dense(kernel_initializer='RandomUniform',input_dim=11,units=6,activation='relu' ))
classifier.add(Dense(kernel_initializer='RandomUniform',units=6,activation='relu' ))
classifier.add(Dense(kernel_initializer='RandomUniform',units=1,activation='sigmoid' ))

classifier.compile(optimizer='adam',loss='binary_crossentropy' , metrics=['accuracy'])


classifier.fit(x_train,y_train,batch_size=10,epochs=100)

y_pred= classifier.predict(x_test)
y_pred= (y_pred>0.5)

from sklearn.metrics import confusion_matrix

cm=confusion_matrix(y_test,y_pred)



y_specific = classifier.predict(np.array([]))