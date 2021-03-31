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


def cost(t, v):
    return np.sum((t - v) ** 2, axis = -1)


def karges(kwargs, check, default):
    if check in kwargs:
        return kwargs[check]
    else:
        return default


def generateWeights(layerData, **kwargs):
    minimum, maximum = karges(kwargs, 'minimum', -1), karges(kwargs, 'maximum', 1)
    return [np.random.uniform(minimum, maximum, [layerData[1:][layer], layerData[layer] + 1]) for layer in range(len(layerData) - 1)]


def layer(inputs, weights, **kwargs):
    return karges(kwargs, 'actFunc', sigmoid)(np.tensordot(np.append(inputs, np.full(list(inputs.shape)[:-1] + [1], 1), axis = -1), weights, axes = [[-1], [-1]]))


def neuralNetwork(inputs, weights, **kwargs):
    actFunc, finalActFunc = karges(kwargs, 'actFunc', sigmoid), karges(kwargs, 'finalActFunc', softmax)
    neuronOutputs = []
    layerInputs = deepcopy(inputs)
    for layerWeights in weights[:-1]:
        layerInputs = layer(layerInputs, layerWeights, actFunc = actFunc)
        neuronOutputs.append(deepcopy(layerInputs))
    
    neuronOutputs.append(layer(layerInputs, weights[-1], actFunc = finalActFunc))
    return neuronOutputs


def generateDx(inputs, weights, **kwargs):
    dx, neuronAmount, weightAmount = karges(kwargs, 'dx', 0.01), karges(kwargs, 'neuronAmount', len(weights)), karges(kwargs, 'weightAmount', len(weights[0]))
    totalWeightAmount = neuronAmount * weightAmount
    return np.identity(len(inputs)) * dx + inputs, np.reshape(np.tile(np.append(dx, np.zeros((totalWeightAmount))), totalWeightAmount)[:totalWeightAmount ** 2], (neuronAmount, weightAmount, neuronAmount, weightAmount)) + weights


def optimizer(inputs, weights, outputs, **kwargs):
    rawCost, learningRate, dx = karges(kwargs, 'rawCost', cost(outputs, layer(inputs, weights, **kwargs))), karges(kwargs, 'learningRate', 0.1), karges(kwargs, 'dx', 0.01)
    dxInputs, dxWeights = generateDx(inputs, weights, **kwargs)
    return inputs - (cost(outputs, layer(dxInputs, weights, **kwargs)) - rawCost) / dx * learningRate, weights - (cost(outputs, layer(inputs, dxWeights, **kwargs)) - rawCost) / dx * learningRate


def backProp(inputs, weights, outputs, **kwargs):
    layerInputs = neuralNetwork(inputs, weights, **kwargs)[:-1][::-1] + [inputs]
    newWeights = deepcopy(weights)[::-1]
    targetOutputs = outputs
    for layerIndex, layerWeights in enumerate(newWeights):
        targetOutputs, newWeights[layerIndex] = optimizer(layerInputs[layerIndex], layerWeights, targetOutputs, **kwargs)
    return newWeights[::-1]