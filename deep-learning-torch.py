# -*- coding: utf-8 -*-
"""
Created on Tue Sep 24 23:03:37 2019

@author: dcamp
"""

import torch 

x = torch.tensor(-3., requires_grad=True)
y = torch.tensor(5., requires_grad=True)
z = torch.tensor(-7., requires_grad=True)

q = x + y
f = q * z

f.backward()

# Derivitives with respect to themselves in the f equation (line 15) are calculated for each variable
print('Gradient of z is: ' + str(z.grad))
print('Gradient of y is: ' + str(y.grad))
print('Gradient of x is: ' + str(x.grad))


###############################################

# Your input will be images of size (28, 28), so images containing 784 pixels. 
# Your network will contain an input_layer (provided for you), a hidden layer 
# with 200 units, and an output layer with 10 classes. 

input_layer = torch.rand(784)

# Initialize the weights of the neural network
weight_1 = torch.rand(784, 200)
weight_2 = torch.rand(200, 10)

# Multiply input_layer with weight_1
hidden_1 = torch.matmul(input_layer, weight_1)

# Multiply hidden_1 with weight_2
output_layer = torch.matmul(hidden_1, weight_2)
print(output_layer)

print('object oriented version'.upper())
from torch import nn

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        
        # Instantiate all 2 linear layers  
        self.fc1 = nn.Linear(784, 200)
        self.fc2 = nn.Linear(200, 10)

    def forward(self, x):
      
        # Use the instantiated layers and return x
        x = self.fc1(x)
        x = self.fc2(x)
        return x
 
print('Example with ReLU activation function')
# 4 inputs 6 nodes in a single hidden layer and 2 outputs  
input_layer = torch.rand(4)

# Instantiate ReLU activation function as relu
relu = nn.ReLU()

# Initialize weight_1 and weight_2 with random numbers
weight_1 = torch.rand(4, 6)
weight_2 = torch.rand(6, 2)

# Multiply input_layer with weight_1
hidden_1 = torch.matmul(input_layer, weight_1)

# Apply ReLU activation function over hidden_1 and multiply with weight_2
hidden_1_activated = relu(hidden_1)
print(torch.matmul(hidden_1_activated, weight_2))
