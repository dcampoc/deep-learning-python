# -*- coding: utf-8 -*-
"""
Created on Wed Sep 25 15:26:07 2019

@author: damian.campo
"""

import torch 
#pip install torchvision==0.1.8
# conda install torchvision -c pytorch
import torchvision
import torch.utils.data
import torchvision.transforms as transforms

#   Mean and stadard deviation of each channel
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.4914, 0.48216, 0,44653),
                                                    (0.24703, 0.24349, 0.26159))])

#   'transform' allows to transform images into tensors
trainset = torchvision.dataset.CIFAR10(root='./data', train=True, download=True, transform=transform)
testset = torchvision.dataset.CIFAR10(root='./data', train=False, download=True, transform=transform)

trainloader = torch.utils.data.DataLoader(trainset, batch_size=32, shuffle=True, num_workers=4)
testloader = torch.utils.data.DataLoader(testset, batch_size=32, shuffle=False, num_workers=4)

# Compute the shape of the training set and testing set
trainset_shape = trainloader.dataset.train_data.shape
testset_shape = testloader.dataset.test_data.shape

# Print the computed shapes
print(trainset_shape, testset_shape)

# Compute the size of the minibatch for training set and testing set
trainset_batchsize = trainloader.batch_size
testset_batchsize = testloader.batch_size

# Print sizes of the minibatch
print(trainset_batchsize, testset_batchsize)