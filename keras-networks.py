# -*- coding: utf-8 -*-
"""
Created on Sat Sep 14 16:03:57 2019

@author: dcamp
"""

# pip install Keras (install keras)
# pip install tensorflow==1.14.0 (install tensor flow compatible with keras) 
# pip install --upgrade tensorflow
import os
import pandas as pd
from keras.layers import Dense
from keras.models import Sequential
from keras.models import load_model

os.chdir(r'C:\Users\dcamp\Documents\python-practice\Deep learning')
print('first exersize'.upper())
df = pd.read_csv('hourly_wages.csv')
print(df.head())

predictors = df.drop(['wage_per_hour'], axis=1).values
target = df['wage_per_hour'].values
# Save the number of columns in predictors: n_cols
n_cols = predictors.shape[1]

# Set up the model: model
model = Sequential()

# Add the first layer (50 nodes and it is indicated that it takes n_cols number of features 
# and any number of samples with such features [specified as the empty space in the touple])
model.add(Dense(50, activation='relu', input_shape=(n_cols,)))

# Add the second layer (32 layers ì)
model.add(Dense(32, activation='relu'))

# Add the output layer (single layer)
model.add(Dense(1))

model.compile(optimizer='adam', loss='mean_squared_error', metrics=['accuracy'])
model.fit(predictors, target, epochs=10)
model.save('model_file_1.h5')

###################################################################

print('second exersize'.upper())
from keras.utils import to_categorical
from keras.callbacks import EarlyStopping
df = pd.read_csv('titanic_all_numeric.csv')
print(df.head())

# Convert the target to categorical: target
target = to_categorical(df.survived)
predictors = df.drop(['survived'], axis=1).values
n_cols = predictors.shape[1]

# Set up the model
model_1 = Sequential()

# Add the first layer
model_1.add(Dense(32, activation='relu',input_shape=(n_cols,)))

# Add the output layer
model_1.add(Dense(2, activation='softmax'))

# Compile the model
model_1.compile(optimizer='sgd', loss='categorical_crossentropy', metrics=['accuracy'])

# patiance indicates the number of epochs the model can go without improvement before we stop training (2 or 3 are good values)
early_stopping_monitor = EarlyStopping(patience=2)

# Fit the model (validation_split encodes the fraction of the data used for validation)
model_1_training = model_1.fit(predictors, target, epochs=20, validation_split=0.3, callbacks=[early_stopping_monitor], verbose=False)
#######################################################
import matplotlib.pyplot as plt

# Create the new model: model_2
model_2 = Sequential()

# Add the first and second layers
model_2.add(Dense(100, activation='relu', input_shape=(n_cols,)))
model_2.add(Dense(100, activation='relu'))

# Add the output layer
model_2.add(Dense(2, activation='softmax'))

# Compile model_2
model_2.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Fit model_2
model_2_training = model_2.fit(predictors, target, epochs=20, validation_split=0.3, callbacks=[early_stopping_monitor], verbose=False)

# Create the plot
plt.plot(model_1_training.history['val_loss'], 'r', model_2_training.history['val_loss'], 'b')
plt.xlabel('Epochs')
plt.ylabel('Validation loss')
plt.grid()
plt.show()

plt.figure()
plt.plot(model_1_training.history['val_acc'], 'r', model_2_training.history['val_acc'], 'b')
plt.xlabel('Epochs')
plt.ylabel('Validation accuracy')
plt.grid()
plt.show()


#######################
print('load and predict with an already saved network')
my_model = load_model('model_file_1.h5')
df = pd.read_csv('hourly_wages.csv')
data_to_predict_with = df.drop(['wage_per_hour'], axis=1).values
target = df['wage_per_hour'].values
target = target.reshape(-1,1)
predictions = my_model.predict(data_to_predict_with)
df_out = pd.DataFrame({'target': list(target), 'predictions': list(predictions)})
print(df_out)


##################################################################




