'''
Saywer 
06/09/2023
First demo neuronal network using Keras
'''
from keras import models
from keras import layers
from keras.datasets import mnist
from keras.utils import to_categorical

# Preparing datasets (tensors) for training and testing the neuronal network
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

# Building a sequential model and adding 2 dense layers
network = models.Sequential()
network.add(layers.Dense(512, activation='relu', input_shape=(28 * 28,)))
network.add(layers.Dense(10, activation='softmax'))

# Compilation step: defining optimizer, loss function and metric
network.compile(optimizer='rmsprop',
loss='categorical_crossentropy',
metrics=['accuracy'])

# Reshaping the input array as the neuronal network expects 
train_images = train_images.reshape((60000, 28 * 28))
train_images = train_images.astype('float32') / 255
test_images = test_images.reshape((10000, 28 * 28))
test_images = test_images.astype('float32') / 255

# Preparing the labels 
train_labels = to_categorical(train_labels)
test_labels = to_categorical(test_labels)

# Network training: fitting the model to its training data
network.fit(train_images, train_labels, epochs=5, batch_size=128)

# Testing our trained neural network 
test_loss, test_acc = network.evaluate(test_images, test_labels)
print('test_acc:', test_acc)