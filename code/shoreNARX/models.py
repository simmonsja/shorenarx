import torch.nn as nn
import numpy as np
from  collections import OrderedDict

################################################################################
################################################################################

def NARX_NN(inputsNum, layers, layerSize, dropout):
    '''
    Simple nn model with variable initial layer size and number of layers.
    Halves the number of neurons each layer with min layer size of 12
    
    Parameters:
    - inputsNum (int): Number of input features.
    - layers (int): Number of hidden layers.
    - layerSize (int): Initial size of the hidden layers.
    - dropout (float): Dropout rate for regularization.
    
    Returns:
    - nn.Sequential: Sequential neural network model.
    '''
    #setup the layers
    layerList = []
    for ii in range(layers):
        layerList.append(layerSize)
        layerSize = int(np.max([layerSize/2,12]))
    listOD = []
    #account for the input to first hidden layer
    prevLayer = inputsNum
    num = 1
    #do the hidden layers
    for thisLay in layerList:
        listOD.append(('linear' + num.__str__(),nn.Linear(prevLayer, thisLay)))
        listOD.append(('relu' + num.__str__(),nn.ReLU()))
        listOD.append(('dropout' + num.__str__(),nn.Dropout(p=dropout)))
        prevLayer = thisLay
        num += 1
    #and now the last layer
    listOD.append(('linear' + num.__str__(),nn.Linear(prevLayer, 1)))
    return nn.Sequential(OrderedDict(listOD))

################################################################################
################################################################################
