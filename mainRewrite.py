from PIL import Image
import time
import _pickle
import random; random.seed(1)
import numpy as np
import NNrewriteV5 as nn

if __name__ == '__main__':
	def grabRGB(uuid, path, size):
		f = Image.open(f'{path}{uuid}.tif').resize(size[:-1])
		a = np.array(f)
		return np.ndarray.flatten(a)/255

	def grabGray(uuid, path, size):
		f = Image.open(f'{path}{uuid}.tif').resize(size)
		a = np.array([[sum(col)/765 for col in row] for row in np.array(f)])
		return np.reshape(a, size[0]*size[1])

	def deepcopy(x):
		return _pickle.loads(_pickle.dumps(x))

	print('Imports Sucessfull')

	path = 'D:\\Users\\Koral Kulacoglu\\Coding\\python\\AI\\YRSTF2021\\Data\\'
	labelsFile = open(f'{path}train_labels.csv')
	labels = [row.split(',') for row in labelsFile.read().split('\n')][1:]
	labelsFile.close()

	print('Data Read')

	dx = 1.001
	rate = 0.1
	beta = 0.9
	scale = 0.1
	layerData = [18, 18, 1]
	colors = 1
	resultPer = 1

	if colors == 1:
		grab = grabGray
		size = (96, 96)
	
	else:
		grab = grabRGB
		size = (96, 96, 3)

	weights = nn.generateWeights(layerData, size[0]*size[1]*colors)

	print('Variables Initialized')
	open(f'{path}scores.csv', 'w+').truncate(0)

	print('Training...')

	start = time.time()

	num, cost = 1, 0
	for row in labels:
		inputs = grabGray(row[0], f'{path}train\\', size)
		outputs = [int(row[1])]
		
		inps = nn.neuralNetwork(inputs, weights)
		networkInputs = inps[-2::-1] + deepcopy(inputs)

		for layer in range(len(layerData)):
			print(networkInputs)
			weights[layer], networkInputs[layer] = nn.backPropagation(networkInputs[layer], weights[layer], outputs, dx, rate=rate, beta=beta, scale=scale)
	
		np.save(f'{path}multiWeights.npy', np.array(weights, dtype=object))
		iterCost, iterResults = nn.neuralNetworkCost(inputs, weights, outputs)
		cost += iterCost

		if not num%resultPer:
			print('\n\nNetwork:', num)
			print(f'Time: {time.time() - start}s')
			print('Cost:', cost/resultPer)
			print('\nPred:', iterResults)
			print('Real:', outputs)

			f = open(f'{path}scores.csv', 'a')
			f.write(f'\n{cost/resultPer}'); f.close()
			cost = 0
		
		num += 1
