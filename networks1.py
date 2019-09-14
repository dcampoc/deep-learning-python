# -*- coding: utf-8 -*-
"""
Created on Fri Sep 13 23:23:36 2019

@author: dcamp
"""

import numpy as np
import os
import matplotlib.pyplot as plt
# Set directory where the information is 
os. chdir(r"C:\Users\dcamp\Documents\python-practice\Deep learning")
#   Definition of th activation cuntion ReLU (it is 0 qwhen values are negtive and the value itself when it is positive)
def relu(input):
    '''Define your relu activation function here'''
    # Calculate the value for the output of the relu function: output
    output = max(input, 0)
    
    # Return the value just calculated
    return(output)

print('Exercise 1'.upper())
# Consider a ANN with a single layer of two nodes(0 and 1). Let us consider two inputs and a single output
print('forward propagation in a small neural networks'.upper())
input_data = np.array([2, 3])
weights = {'node_0': np.array([1, 1]),
           'node_1': np.array([-1, 1]),
           'output': np.array([2, -1])}
# Inputs/outputs of the nodes in the hidden layer, tanh() is considered as an activation function
node_0_input = (input_data * weights['node_0']).sum()
node_0_output_tan = np.tanh(node_0_input)
node_0_output_ReLU = relu(node_0_input)

node_1_input = (input_data * weights['node_1']).sum()
node_1_output_tan = np.tanh(node_1_input)
node_1_output_ReLU = relu(node_1_input)


hidden_layer_values_ReLU = np.array([node_0_output_ReLU, node_1_output_ReLU])
hidden_layer_values_tan = np.array([node_0_output_tan, node_1_output_tan])

print('values of the nodes in the hidden layer (ReLU and tanh):'.upper())
print(hidden_layer_values_ReLU)
print(hidden_layer_values_tan)

print('value of the single output (ReLU and tanh):'.upper())
output_ReLU = (hidden_layer_values_ReLU*weights['output']).sum()
output_ReLU = relu(output_ReLU)
output_tanh = (hidden_layer_values_tan*weights['output']).sum()
output_tanh = np.tanh(output_tanh)
print(output_ReLU)
print(output_tanh)


###################### Exercise two: two weights going directly to the output node
print('Exercise 2'.upper())
print('updating weights'.upper())

weights = np.array([1, 2])
input_data = np.array([3, 4])
target = 6 
learning_rate = 0.01
eror_hist = []

for i in list(range(10)):
    preds = (weights * input_data).sum()
    error = preds - target
    print('Error:', error)

    gradient = 2*input_data * error
    weights = weights - learning_rate*gradient
    eror_hist.append(error)
print('Final weights:', weights)

epochs = 1 +  np.array(list(range(10)))
plt.plot(epochs, eror_hist,c='red', marker='D', alpha=0.8)
plt.xlabel('epochs')
plt.ylabel('error')
plt.grid()





