from copy import deepcopy
import time
import random; random.seed(1)
import numpy as np
from PIL import Image
import NeuralNetwork as nn
import Debounce as D
print('Imports Sucessfull')

def grab(uuid, path):
	f = Image.open(f'{path}{uuid}.tif')
	a = np.array(f)
	return [1-row/255 for row in np.reshape(a, (27648))]

path = 'D:\\Users\\Koral Kulacoglu\\Coding\\python\\AI\\YRSTF2021\\Data\\'
labelsFile = open(f'{path}train_labels.csv')
labels = [row.split(',') for row in labelsFile.read().split('\n')][1:]
labelsFile.close()

print('Data Read')

dx = 0.001
rate = 0.1
beta = 0.9
scale = 0.1
layerData = [32, 32, 32, len(labels[0][1])]
weights = nn.generateWeights(layerData, 27648)

print('Variables Initialized')
open(f'{path}scores.csv', 'w+').truncate(0)

print('Training...')
start = time.time()

num, cost = 0, 0
for row in labels:
	inputs = grab(row[0], f'{path}train\\')
	outputs = int(row[1])

	weights, newOutputs = D.backPropagation(inputs, weights, outputs, dx, rate, beta, scale)

	cost += nn.neuralNetworkCost(inputs, weights, outputs)
	t = int(time.time() - start)

	if not (inp+1)%10:
		print('\n\nNetwork:', num+1)
		print(f'Time: {t}s')
		print('Cost:', cost/10)
		print('\nPred:', newOutputs)
		print('Real:', outputs)

		f = open(f'{path}scores.csv', 'a')
		f.write(f'\n{cost/10}'); f.close()
		cost = 0

	np.save(f'{path}weights.npy', np.array(weights, dtype=object))
	num += 1
