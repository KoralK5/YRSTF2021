import numpy as np
import _pickle
from numba import jit

def deepcopy(x):
	return _pickle.loads(_pickle.dumps(x))

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
	
	shape = list(inputs.shape)
	shape[-1] = 1
	bias = np.full(shape, 1)

	return actFunc(np.append(inputs, bias, axis = len(shape) - 1) @ weights.transpose())

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
	neuronOutputs.append(weights[-1])
	return neuronOutputs

def optimizer(inputs, weights, outputs, dx, **kwargs):
	if 'actFunc' in kwargs:
		actFunc = kwargs['actFunc']
	else:
		actFunc = sigmoid
	if 'normalCost' in kwargs:
		normalCost = kwargs['normalCost']
	else:
		normalCost = (outputs - layer(inputs, weights, actFunc = actFunc)) ** 2
	if 'rate' in kwargs:
		rate = kwargs['rate']
	else:
		rate = 0.1

	neuronAmount, weightAmount = len(weights), len(weights[0])
	totalWeightAmount = neuronAmount * weightAmount
	
	dxWeights = np.reshape(np.tile(np.append(dx, np.zeros((totalWeightAmount))), totalWeightAmount)[:totalWeightAmount ** 2], (neuronAmount, weightAmount, neuronAmount, weightAmount)) + weights
	dxInputs = np.identity(len(inputs)) * dx + inputs

	dxWeightsOutputs, dxInputsOutputs = layer(inputs, dxWeights, actFunc=actFunc), layer(dxInputs, weights, actFunc=actFunc)

	dxWeightsCosts, dxInputsCosts = (outputs - dxWeightsOutputs) ** 2, (outputs - dxInputsOutputs) ** 2
	weightsGradients, inputsGradients = (dxWeightsCosts - normalCost) / dx, (dxInputsCosts - normalCost) / dx
	
	print(weights)

	return weights - weightsGradients * rate, inputs - inputsGradients * rate

def backPropagation(inputs, weights, outputs, dx, **kwargs):
	newWeights = deepcopy(weights)[::-1]
	newOutputs = deepcopy(outputs)
	
	inps = neuralNetwork(inputs, weights)
	networkInputs = inps[-2::-1] + [deepcopy(inputs)]

	for currentLayer in range(len(networkInputs)):
		newWeights[currentLayer], newOutputs = optimizer(networkInputs[currentLayer], newWeights[currentLayer], newOutputs, dx, **kwargs)

	return newWeights[::-1], newOutputs, np.sum((outputs - newOutputs) ** 2)
