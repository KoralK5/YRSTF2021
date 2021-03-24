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


def debounce(inputs, weights, outputs, dx, **kwargs):
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
    dxWeightsOutputs = layer(inputs, dxWeights, actFunc = actFunc)
    dxInputsOutputs = layer(dxInputs, weights, actFunc = actFunc)
    dxWeightsCosts = (outputs - dxWeightsOutputs) ** 2
    dxInputsCosts = (outputs - dxInputsOutputs) ** 2
    # note find a way to possible make shape of costs the same as weights (possibly adjust generation method, too sleepy rn tho)
