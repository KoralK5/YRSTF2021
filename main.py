from PIL import Image
import time
import _pickle
import numpy as np
import cupy as cp
import xdNNrewrite as nn
import time

f = open('/home/iantitor/Desktop/histopathologic-cancer-detection/train_labels.csv', 'r')
data = f.read().split('\n')[1:-1]
f.close()
for x, d in enumerate(data):
    a = d.split(',')
    data[x] = [a[0], np.array([int(a[1])])]
print('Dataset list initialized successfully')


def grabRGB(uuid, path):
	f = Image.open(f'{path}{uuid}.tif')
	a = np.array(f)
	return np.ndarray.flatten(a)/255


def datasetFunc(index, **kwargs):
    labels, path = kwargs['labels'], kwargs['path']
    inputs, outputs = labels[index % len(labels)]
    inputs = grabRGB(inputs, path)
    return inputs, outputs

weights = nn.generateWeights([27648, 64, 32, 16, 1], minimum = -1, maximum = 1)

start = time.time()
newWeights = nn.train(datasetFunc, weights, iterLimit = len(data), path = '/home/iantitor/Desktop/histopathologic-cancer-detection/train/', labels = data, costThreshold = -1, learningRate = 1)
print(time.time() - start)
