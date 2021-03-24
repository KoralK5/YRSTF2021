import numpy as np
import cupy as cp
import _pickle as pkl
from numba import jit

@jit(nopython=True)
def softmax(x):
	return np.exp(x - np.max(x)) / np.exp(x - np.max(x)).sum()

@jit(nopython=True)
def sigmoid(x):
	return 1 / (1 + np.exp(-x))

def generateWeights(layerData, inputQuantity):
	layerDepthLimit = len(layerData)
	augmentedLayerData = [inputQuantity] + pkl.loads(pkl.dumps(layerData))
	return [np.random.rand(layerData[layerDepth], augmentedLayerData[layerDepth] + 1) - 0.5 for layerDepth in range(layerDepthLimit)]

def layer(inputs, weights, **kwargs):
	if 'actFunc' in kwargs:
		actFunc = kwargs['actFunc']
	else:
		actFunc = sigmoid
	return actFunc(np.append(inputs, 1) @ weights.transpose())

def neuralNetwork(inputs, weights, **kwargs):
	if 'actFunc' in kwargs:
		actFunc = kwargs['actFunc']
	else:
		actFunc = sigmoid
	if 'lastActFunc' in kwargs:
		lastActFunc = kwargs['lastActFunc']
	else:
		lastActFunc = softmax
	neuronOutputs = []
	layerInputs = pkl.loads(pkl.dumps(inputs))
	for layerWeights in weights[:-1]:
		layerInputs = layer(layerInputs, layerWeights, actFunc = actFunc)
		neuronOutputs.append(pkl.loads(pkl.dumps(layerInputs)))
	neuronOutputs.append(layerInputs, weights[-1], actFunc = lastActFunc)
	return neuronOutputs

def layerCost(inputs, weights, outputs):
	return np.sum((outputs - layer(inputs, weights)) ** 2)

def neuralNetworkCost(inputs, weights, outputs):
	return np.sum((outputs - neuralNetwork(inputs, weights)[-1]) ** 2)

def adjustNeuron(self, neuron):
	costF = nn.layerCost(inputs, weights, outputs)
	newInputs[neuron] += dx
	costS = nn.layerCost(newInputs, weights, outputs)
	return (costS - costF) / dx

def adjustWeight(self, neuron, weight):
	costF = layerCost(inputs, weights, outputs)
	newWeights[neuron][weight] += dx
	costS = layerCost(inputs, newWeights, outputs)
	return (costS - costF) / dx

def adjustLayer(dx, inputs, weights, outputs, **kwargs):
	actFunc = np.vectorize(adjustWeight)
	#parallel processing
	pass

def debounce(inputs, weights, outputs, dx, **kwargs):
	if 'actFunc' in kwargs:
		actFunc = kwargs['actFunc']
	else:
		actFunc = sigmoid
	if 'lastActFunc' in kwargs:
		lastActFunc = kwargs['lastActFunc']
	else:
		lastActFunc = softmax
	neuronAmount = len(weights)
	weightAmount = len(weights[0])
	totalWeightAmount = neuronAmount * weightAmount
	dxWeights = np.reshape(np.tile(np.append(dx, np.zeros((totalWeightAmount))), totalWeightAmount)[:totalWeightAmount ** 2], (totalWeightAmount, neuronAmount, weightAmount)) + weights

	inps = neuralNetwork(inputs, weights)

	inputs = inps[-2::-1] + [inputs]

	weights = weights[::-1]
	for layer in range(weightAmount):
		weights[layer], outputs = adjustLayer(dx, **kwargs)
	
	return weights[::-1], inps[-1]
