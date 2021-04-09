print('numpy'); import numpy as np
print('matplotlib.pyplot'); import matplotlib.pyplot as plt
print('IPython.display'); from IPython.display import clear_output
print('Image'); from PIL import Image
print('tensorflow'); import tensorflow as tf

print('Imports Sucessfull')

path = 'enter your path here'
labelsFile = open(f'{path}train_labels.csv')
labels = [row.split(',') for row in labelsFile.read().split('\n')][1:]
labelsFile.close()

print('Hyperparameters Set')

def testRGB(uuid):
    image = Image.open(f'D:\\Users\\Koral Kulacoglu\\Coding\\python\\AI\\YRSTF2021\\Data\\test\\{uuid[0]}.tif')
    return np.array(image).reshape(-1, 96, 96, 3)/255.0, image

model = tf.keras.models.load_model('HistoCNN.model')

labelsFile = open(f'{path}sample_submission.csv')
labels = [row.split(',') for row in labelsFile.read().split('\n')][1:]
labelsFile.close()

print('Testing Ready')

categories = ['Benign', 'Malignant']

inputs0, image0 = testRGB(labels[0])
inputs1, image1 = testRGB(labels[1])

pred0 = model.predict(inputs0)
pred1 = model.predict(inputs1)

print('   PREDICTION :', categories[round(pred0[0][0])], '-', format(pred0[0][0], '.2f'))

plt.imshow(image0)
plt.show(block=False)

print('   PREDICTION :', categories[round(pred1[0][0])], '-', format(pred1[0][0], '.2f'))

plt.imshow(image1)
plt.show(block=False)

from random import randrange
from IPython.display import clear_output

def show():
    categories = ['Benign', 'Malignant']

    curr = 0
    while curr != 100:
        loc = randrange(0, len(labels))
        inputs, image = testRGB(labels[loc])

        pred = model.predict(inputs)

        print('   PREDICTION :', categories[round(pred[0][0])], '-', format(pred[0][0], '.2f'))

        plt.imshow(image)
        plt.show(block=False)
        plt.pause(2)
        plt.close()
        clear_output(wait = True)

        curr += 1
        
show()
