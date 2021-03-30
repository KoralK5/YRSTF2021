import numpy as np
import _pickle as pkl
from numba import jit


@jit(nopython=True)
def softmax(x):
	return np.exp(x - np.max(x)) / np.exp(x - np.max(x)).sum()


@jit(nopython=True)
def sigmoid(x):
	return 1 / (1 + np.exp(-x))

def deepcopy(x):
	return pkl.loads(pkl.dumps(x))


def generateWeights(layerData, inputQuantity, minimum = -1, maximum = 1):
	layerDepthLimit = len(layerData)
	augmentedLayerData = [inputQuantity] + deepcopy(layerData)
	return [np.random.uniform(minimum, maximum, [layerData[layerDepth], augmentedLayerData[layerDepth] + 1]) - 0.5 for layerDepth in range(layerDepthLimit)]


def layer(inputs, weights, **kwargs):
	if 'actFunc' in kwargs:
		actFunc = kwargs['actFunc']
	else:
		actFunc = sigmoid
	shape = list(inputs.shape)[:-1] + [1]
	biases = np.full(shape, 1)
	outputs = actFunc(np.append(inputs, biases, axis = len(shape) - 1) @ weights.transpose(np.roll(np.arange(len(weights.shape)), -1)))
	return outputs.transpose(np.roll(np.arange(len(outputs.shape)), 1))


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


def dxMaker(weights, dx):
	neuronAmount, weightAmount = len(weights), len(weights[0])
	totalWeightAmount = neuronAmount * weightAmount
	return np.reshape(np.tile(np.append(dx, np.zeros((totalWeightAmount))), totalWeightAmount)[:totalWeightAmount ** 2], (neuronAmount, weightAmount, neuronAmount, weightAmount)) + weights


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
		
	dxWeights = dxMaker(weights, dx)
	dxInputs = np.identity(len(inputs)) * dx + inputs
	dxWeightsOutputs = layer(inputs, dxWeights, actFunc = actFunc)
	dxInputsOutputs = layer(dxInputs, weights, actFunc = actFunc)
	dxWeightsCosts = (outputs - dxWeightsOutputs) ** 2
	dxInputsCosts = (outputs - dxInputsOutputs) ** 2

	weightsGradients, inputsGradients = np.transpose((dxWeightsCosts - normalCost) / dx)[0], np.transpose((dxInputsCosts - normalCost) / dx)[0]

	return [weights[0] - [row[0] for row in weightsGradients * rate]], inputs - inputsGradients * rate


def backPropagation(inputs, weights, outputs, dx, **kwargs):
	newWeights = deepcopy(weights)[::-1]
	newOutputs = deepcopy(outputs)
	
	inps = neuralNetwork(inputs, weights)
	networkInputs = inps[-2::-1] + [deepcopy(inputs)]

	for currentLayer in range(len(networkInputs)):
		print('case')
		newWeights[currentLayer], networkInputs[currentLayer] = optimizer(networkInputs[currentLayer], newWeights[currentLayer], newOutputs, dx, **kwargs)

	return newWeights[::-1]
