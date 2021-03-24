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

def deepcopy(x):
	return pkl.loads(pkl.dumps(x))


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


def debounce(inputs, weights, outputs, dx, **kwargs):
	if 'actFunc' in kwargs:
		actFunc = kwargs['actFunc']
	else:
		actFunc = sigmoid
	if 'normalCost' in kwargs:
		normalCost = kwargs['normalCost']
	else:
		normalCost = (outputs - layer(inputs, weights, actFunc = actFunc)) ** 2
	if 'weightVelocity' in kwargs:
		weightVelocity = kwargs['weightVelocity']
	else:
		weightVelocity = deepcopy(weights) * 0
	if 'weightPreVelocity' in kwargs:
		weightPreVelocity = kwargs['weightPreVelocity']
	else:
		weightPreVelocity = deepcopy(weights) * 0
	if 'inputVelocity' in kwargs:
		inputVelocity = kwargs['inputVelocity']
	else:
		inputVelocity = deepcopy(inputs) * 0
	if 'inputPreVelocity' in kwargs:
		inputPreVelocity = kwargs['inputPreVelocity']
	else:
		inputPreVelocity = deepcopy(inputs) * 0
	if 'learningRate' in kwargs:
		learningRate = kwargs['learningRate']
	else:
		learningRate = 1
	if 'weightScale' in kwargs:
		weightScale = kwargs['weightScale']
	else:
		weightScale = 1
	if 'inputScale' in kwargs:
		inputScale = kwargs['inputScale']
	else:
		inputScale = 1
	if 'weightBeta' in kwargs:
		weightBeta = kwargs['weightBeta']
	else:
		weightBeta = 1
	if 'inputBeta' in kwargs:
		inputBeta = kwargs['inputBeta']
	else:
		inputBeta = 1
	neuronAmount, weightAmount = len(weights), len(weights[0])
	totalWeightAmount = neuronAmount * weightAmount
	dxWeights = np.reshape(np.tile(np.append(dx, np.zeros((totalWeightAmount))), totalWeightAmount)[:totalWeightAmount ** 2], (neuronAmount, weightAmount, neuronAmount, weightAmount)) + weights
	dxInputs = np.identity(len(inputs)) * dx + inputs
	dxWeightsOutputs, dxInputsOutputs = layer(inputs, dxWeights, actFunc = actFunc), layer(dxInputs, weights, actFunc = actFunc)
	dxWeightsCosts, dxInputsCosts = (outputs - dxWeightsOutputs) ** 2, (outputs - dxInputsOutputs) ** 2
	dxweightsCosts, inputsGradients = (dxWeightsCosts - normalCost) / dx, (dxInputsCosts - normalCost) / dx
	weightVelocity, inputVelocity = (weightBeta - weightScale * np.tanh(weightVelocity - weightPreVelocity)) * weightVelocity + dxweightsCosts * learningRate, (inputBeta - inputScale * np.tanh(inputVelocity - inputPreVelocity)) * inputVelocity + inputsGradients * learningRate
	newWeights, newInputs = weights - (weightBeta - weightScale * np.tanh(weightVelocity - weightPreVelocity)) * weightVelocity + learningRate * dxweightsCosts, inputs - (inputBeta - inputScale * np.tanh(inputVelocity - inputPreVelocity)) * inputVelocity + learningRate * inputsGradients
	return newWeights, newInputs, 
