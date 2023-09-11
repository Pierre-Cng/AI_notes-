'''
Saywer 
06/09/2023
imbd example with overfitting demo
'''
import numpy as np
import matplotlib.pyplot as plt
from keras import models
from keras import layers
from keras.datasets import imdb

# Preparing datasets (tensors) for training and testing the neuronal network
(train_data, train_labels), (test_data, test_labels) = imdb.load_data(num_words=10000)

'''
train_data[0] # -> [1, 14, 22, 16, ... 178, 32]
train_labels[0] # -> 1

max([max(sequence) for sequence in train_data]) # -> 9999

word_index = imdb.get_word_index()
reverse_word_index = dict([(value, key) for (key, value) in word_index.items()])
decoded_review = ' '.join([reverse_word_index.get(i - 3, '?') for i in train_data[0]])
'''

# Vectorize the word sequences 
def vectorize_sequences(sequences, dimension=10000):
    results = np.zeros((len(sequences), dimension))
    for i, sequence in enumerate(sequences):
        results[i, sequence] = 1.
        return results
    
x_train = vectorize_sequences(train_data)
x_test = vectorize_sequences(test_data)

# x_train[0] # -> array([ 0., 1., 1., ..., 0., 0., 0.])

# Vectorize labels
y_train = np.asarray(train_labels).astype('float32')
y_test = np.asarray(test_labels).astype('float32')


# Building a sequential model and adding dense layers
model = models.Sequential()
model.add(layers.Dense(16, activation='relu', input_shape=(10000,)))
model.add(layers.Dense(16, activation='relu'))
model.add(layers.Dense(1, activation='sigmoid'))

# Compilation step: defining optimizer, loss function and metric
model.compile(optimizer='rmsprop',
loss='binary_crossentropy',
metrics=['accuracy'])

'''
# Alternative way to import and use speific optimizer, loss func and metrics
from keras import optimizers
from keras import losses
from keras import metrics

model.compile(optimizer=optimizers.RMSprop(lr=0.001),
loss=losses.binary_crossentropy,
metrics=[metrics.binary_accuracy])
'''

x_val = x_train[:10000]
partial_x_train = x_train[10000:]
y_val = y_train[:10000]
partial_y_train = y_train[10000:]

# Network training: fitting the model to its training data
history = model.fit(partial_x_train, partial_y_train, epochs=20, batch_size=512, validation_data=(x_val, y_val))

# Plotting the training and validation loss
history_dict = history.history
loss_values = history_dict['loss']
val_loss_values = history_dict['val_loss']
acc_values = history_dict['accuracy']
val_acc_values = history_dict['val_accuracy']
epochs = range(1, len(history_dict['accuracy']) + 1)

# Plotting model loss and accuracy with trained data and compare with validation data
fig, ax = plt.subplots(2)
ax[0].plot(epochs, loss_values, 'bo', label='Training loss')
ax[0].plot(epochs, val_loss_values, 'b', label='Validation loss')
ax[0].set_title('Training and validation loss')
ax[0].set_xlabel('Epochs')
ax[0].set_ylabel('Loss')
ax[0].legend()

ax[1].plot(epochs, acc_values, 'bo', label='Training acc')
ax[1].plot(epochs, val_acc_values, 'b', label='Validation acc')
ax[1].set_title('Training and validation accuracy')
ax[1].set_xlabel('Epochs')
ax[1].set_ylabel('Accuracy')
ax[1].legend()

plt.show()
# We can observe overfitting: the model is not performant with new data (validation data)