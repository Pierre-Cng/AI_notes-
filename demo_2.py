'''
Saywer 
06/09/2023
Tensor manipulation
'''
from keras.datasets import mnist
import matplotlib.pyplot as plt

# Preparing datasets (tensors) for training and testing the neuronal network
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

# A tensor is defined by three key attributes:
print('Tensor rank:', train_images.ndim) # displaying the tensor rank
print('Tensor shape:', train_images.shape) # displaying the tensor shape
print('Tensor dtype:', train_images.dtype) # dipslaying the tensor dtype 

# Tensor slicing:

# 1 element:
plt.imshow(train_images[4], cmap=plt.cm.binary)
plt.show()

# from index 10 to 100:
my_slice = train_images[10:100]
print(my_slice.shape)
# equivalent notation: 
my_slice = train_images[10:100, :, :]
print(my_slice.shape)
# equivalent notation:
my_slice = train_images[10:100, 0:28, 0:28]
print(my_slice.shape)