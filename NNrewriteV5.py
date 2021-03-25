import numpy as np
import _pickle as pkl
from numba import jit

def deepcopy(x):
	return pkl.loads(pkl.dumps(x))

@jit(nopython=True)
def softmax(x):
	return np.exp(x - np.max(x)) / np.exp(x - np.max(x)).sum()

@jit(nopython=True)
def sigmoid(x):
	return 1 / (1 + np.exp(-x))

def generateWeights(layerData, inputQuantity):
	layerDepthLimit = len(layerData)
	augmentedLayerData = [inputQuantity] + deepcopy(layerData)
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
	layerInputs = deepcopy(inputs)
	for layerWeights in weights[:-1]:
		layerInputs = layer(layerInputs, layerWeights, actFunc = actFunc)
		neuronOutputs.append(deepcopy(layerInputs))
	neuronOutputs.append(layerInputs, weights[-1], actFunc = lastActFunc)
	return neuronOutputs

def neuralNetworkCost(inputs, weights, outputs):
	return np.sum((outputs - neuralNetwork(inputs, weights)[-1]) ** 2)

def backPropagation(inputs, weights, outputs, dx, **kwargs):
	if 'actFunc' in kwargs:
		actFunc = kwargs['actFunc']
	else:
		actFunc = sigmoid
	if 'normalCost' in kwargs:
		normalCost = kwargs['normalCost']
	else:
		normalCost = (outputs - layer(inputs, weights, actFunc = actFunc)) ** 2

	neuronAmount, weightAmount = len(weights), len(weights[0])
	totalWeightAmount = neuronAmount * weightAmount
	
	dxWeights = np.reshape(np.tile(np.append(dx, np.zeros((totalWeightAmount))), totalWeightAmount)[:totalWeightAmount ** 2], (neuronAmount, weightAmount, neuronAmount, weightAmount)) + weights
	dxInputs = np.identity(len(inputs)) * dx + inputs
	
	dxWeightsOutputs, dxInputsOutputs = layer(inputs, dxWeights, actFunc=actFunc), layer(dxInputs, weights, actFunc=actFunc)

	dxWeightsCosts, dxInputsCosts = (outputs - dxWeightsOutputs) ** 2, (outputs - dxInputsOutputs) ** 2
	weightsGradients, inputsGradients = (dxWeightsCosts - normalCost) / dx, (dxInputsCosts - normalCost) / dx

	return weights - (weightsGradients * rate), inputs - (inputsGradients * rate)
