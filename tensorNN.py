from copy import deepcopy
from PIL import Image
import numpy as np
from os import system
import random; random.seed(1)
import tensorflow as tf
print('Imports Sucessfull')

def grab(uuid, path):
	f = Image.open(f'{path}{uuid}.tif')
	a = np.array([[1-sum(col)/765 for col in row] for row in np.array(f)])
	return a

path = 'D:\\Users\\Koral Kulacoglu\\Coding\\python\\AI\\YRSTF2021\\Data\\'
labelsFile = open(f'{path}train_labels.csv')
labels = [row.split(',') for row in labelsFile.read().split('\n')][1:]
labelsFile.close()

inputs, outputs = [], []
for row in range(len(labels)):
	inputs.append(grab(labels[row][0], f'{path}train\\'))
	outputs.append([int(labels[row][1]), abs(1-int(labels[row][1]))])
	row += 1

	system('cls')
	print('Reading Data')
	print(f'{int(row/len(labels)*100)}% - {row}/{len(labels)}')
	print('⬜'*int(row/len(labels)*10) +'⬛'*(10-int(row/len(labels)*10)))

system('cls')
print('Data Read')

dx = 1.001
rate = 0.1
beta = 0.9
scale = 0.1
size = (96, 96)
layerData = [24, 13, 8, 2]

model = tf.keras.models.Sequential([
	tf.keras.layers.Flatten(input_shape=(size)),
	tf.keras.layers.Dense(128, activation='relu'),
	tf.keras.layers.Dropout(0.2),
	tf.keras.layers.Dense(10)
])
print('Parameters Set')

predictions = model(inputs[:1]).numpy()
tf.nn.softmax(predictions).numpy()
loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

model.compile(optimizer='adam',
              loss=loss_fn,
              metrics=['accuracy'])

model.fit(inputs, outputs, epochs=5)
model.evaluate(inputs,  outputs, verbose=2)
