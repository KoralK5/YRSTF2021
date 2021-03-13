import numpy as np
from copy import deepcopy

def softmax(x):
	return np.exp(x - np.max(x)) / np.exp(x - np.max(x)).sum()

def ReLU(x):
	return max(x, 0)

def sigmoid(x):
	return 1 / (1 + np.exp(-x))

def generateWeights(layerData, inputQuantity):
	layerDepthLimit = len(layerData)
	augmentedLayerData = [inputQuantity] + deepcopy(layerData)
	return [np.random.rand(layerData[layerDepth], augmentedLayerData[layerDepth] + 1) - 0.5 for layerDepth in range(layerDepthLimit)]

def layer(inputs, weights):
	biasedInputs = np.append(np.array(inputs), 1)
	neuronInputs = np.repeat(np.array([biasedInputs]), len(weights), axis = 0)
	weightedInputs = neuronInputs * np.array(weights)
	return np.array(list(map(sigmoid, np.sum(weightedInputs, axis = 1))))

def neuralNetwork(inputs, weights):
	outputs = []
	layerInputs = deepcopy(inputs)
	for layerWeights in weights:
		layerInputs = layer(layerInputs, layerWeights)
		outputs.append(deepcopy(layerInputs))
	return outputs

def layerCost(inputs, weights, outputs):
	return np.sum((outputs - layer(inputs, weights)) ** 2)
		
def neuralNetworkCost(inputs, weights, outputs):
	return np.sum((outputs - neuralNetwork(inputs, weights)[-1]) ** 2)
