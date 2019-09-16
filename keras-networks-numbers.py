# -*- coding: utf-8 -*-
"""
Created on Mon Sep 16 03:14:16 2019

@author: dcamp
"""

import os
import pandas as pd
from keras.layers import Dense
from keras.models import Sequential
from keras.models import load_model
import matplotlib.pyplot as plt
from keras.callbacks import EarlyStopping
import numpy as np
import copy 

os.chdir(r'C:\Users\dcamp\Documents\python-practice\Deep learning')
print('first exersize'.upper())
df = pd.read_csv('mnist.csv')
print(df.head())
X = []
y = []
n_labels = 10
labels = [0]*n_labels
for i in list(range(df.shape[0])):
    X_image = df.iloc[i].values[1:]
    X.append(X_image)
    lab = copy.deepcopy(labels)
    lab[df.iloc[i].values[0]] = 1
    X_image = X_image.reshape([28,-1])
    y.append(lab)
    #plt.imshow(X_image, cmap=plt.get_cmap('gray'))
X_array = np.asarray(X)
y_array = np.asarray(y)

model = Sequential()
# Add the first hidden layer
model.add(Dense(100, activation='relu', input_shape=(X_array.shape[1],)))

# Add the second hidden layer
model.add(Dense(50, activation='relu'))

# Add the output layer
model.add(Dense(n_labels, activation='softmax'))

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
early_stopping_monitor = EarlyStopping(patience=5)
# Fit the model
model.fit(X_array, y_array,epochs=50, validation_split=0.2, callbacks=[early_stopping_monitor])
