import numpy as np
from itertools import product
import concurrent.futures
from copy import deepcopy

def softmax(x):
	return np.exp(x - np.max(x)) / np.exp(x - np.max(x)).sum()

def ReLU(x):
	return max(x, 0)

def generateWeights(layerData, inputQuantity):
	layerDepthLimit = len(layerData)
	augmentedLayerData = [inputQuantity] + deepcopy(layerData)
	return [np.random.rand(layerData[layerDepth], augmentedLayerData[layerDepth] + 1) - 0.5 for layerDepth in range(layerDepthLimit)]

def sigmoid(x):
	return 1 / (1 + np.exp(-np.sum(x)))

def layerMulti(inputs, weights):
	biasedInputs = np.append(np.array(inputs), 1)
	neuronInputs = np.repeat(np.array([biasedInputs]), len(weights), axis = 0)
	weightedInputs = neuronInputs * np.array(weights)

	with concurrent.futures.ThreadPoolExecutor() as executor:
		result = executor.map(sigmoid, weightedInputs)
	
	out = np.array([])
	for row in result:
		out = np.append(out, row)

	return np.array(out)

def layerSingle(inputs, weights):
	biasedInputs = np.append(np.array(inputs), 1)
	neuronInputs = np.repeat(np.array([biasedInputs]), len(weights), axis = 0)
	weightedInputs = neuronInputs * np.array(weights)
	
	out = list(map(sigmoid, weightedInputs))

	return np.array(out)

def neuralNetwork(inputs, weights):
	outputs = []
	layerInputs = deepcopy(inputs)
	for layerWeights in weights:
		layerInputs = layerSingle(layerInputs, layerWeights)
		outputs.append(deepcopy(layerInputs))
	return outputs

def layerCost(inputs, weights, outputs):
	return np.sum((outputs - layerSingle(inputs, weights)) ** 2)
		
def neuralNetworkCost(inputs, weights, outputs):
	return np.sum((outputs - neuralNetwork(inputs, weights)[-1]) ** 2)
