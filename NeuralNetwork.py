import numpy as np
from numba import njit
import _pickle as cPickle

def generateWeights(layerData, inputQuantity):
	layerDepthLimit = len(layerData)
	augmentedLayerData = [inputQuantity] + cPickle.loads(cPickle.dumps(layerData, -1))
	return [np.random.rand(layerData[layerDepth], augmentedLayerData[layerDepth] + 1) - 0.5 for layerDepth in range(layerDepthLimit)]

@njit(nopython=True)
def softmax(x):
	return np.exp(x - np.max(x)) / np.exp(x - np.max(x)).sum()

@njit(nopython=True)
def ReLU(x):
	return max(x, 0)

@njit(nopython=True)
def sigmoid(x):
	return 1 / (1 + np.exp(-x))

def layer(inputs, weights):
	biasedInputs = np.append(np.array(inputs), 1)
	neuronInputs = np.repeat(np.array([biasedInputs]), len(weights), axis = 0)
	
	actFunc = np.vectorize(sigmoid)
	weightedInputs = actFunc(np.sum(neuronInputs * np.array(weights), 1))

	return weightedInputs

def neuralNetwork(inputs, weights):
	outputs = []
	layerInputs = cPickle.loads(cPickle.dumps(inputs, -1))
	for layerWeights in weights:
		layerInputs = layer(layerInputs, layerWeights)
		outputs.append(cPickle.loads(cPickle.dumps(layerInputs, -1)))
	return outputs

def layerCost(inputs, weights, outputs):
	return np.sum((outputs - layer(inputs, weights)) ** 2)

def neuralNetworkCost(inputs, weights, outputs):
	return np.sum((outputs - neuralNetwork(inputs, weights)[-1]) ** 2)
