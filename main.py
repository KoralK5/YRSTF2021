from copy import deepcopy
import time
import random; random.seed(1)
import numpy as np
from PIL import Image
import TNeuralNetwork as nn
import TDebounce as D
print('Imports Sucessfull')

def read(uuid, path):
	f = Image.open(f'{path}{uuid}.tif')
	f.show()
	a = np.array(f)
	return np.reshape(a, (27648))

path = 'D:\\Users\\Koral Kulacoglu\\Coding\\python\\AI\\YRSTF2021\\Data\\'
labelsFile = open(f'{path}train_labels.csv')
labels = [row.split(',') for row in labelsFile.read().split('\n')][1:]
labelsFile.close()

inputsD, outputsD = grab(path, 200, 200)

print('Data Formatted')

dx = 0.001
rate = 0.1
beta = 0.9
scale = 0.1
layerData = [1000, 250, 62, 15, len(outputsD[0])]

weights = nn.generateWeights(layerData, len(inputsD[0]))

print('Weights Initialized')
print('Training...')

open(f'{path}scores.csv', 'r+').truncate(0)
start = time.time()

num, cost, t = 0, 0, 32347
for inp in range(len(outputsD)):
	inputs = inputsD[0]
	outputs = outputsD[0]

	inputsD = inputsD[1:]
	outputsD = outputsD[1:]

	weights, newOutputs = GD.backPropagation(inputs, weights, outputs, dx, rate)

	cost += nn.neuralNetworkCost(inputs, weights, outputs)
	t = int(32347 - time.time() + start)
	if t < 0:
		break

	if not (inp+1)%10:
		print('\n\nNetwork:', num+1)
		print(f'Time: {t}s')
		print('Cost:', cost/10)
		print('\nPred:', newOutputs)
		print('Real:', outputs)

		f = open(f'{path}scores.csv', 'a')
		f.write(f'\n{cost/10}'); f.close()
		cost = 0

	np.save(f'{path}GDweights.npy', np.array(weights, dtype=object))
	num += 1

print('Final Cost:', cost)
print('Iterations:', num)
