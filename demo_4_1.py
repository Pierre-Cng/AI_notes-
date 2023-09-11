'''
Saywer 
06/09/2023
example of single-label, multiclass classification.
'''
import matplotlib.pyplot as plt
import numpy as np
from keras import models, layers 
from keras.datasets import reuters
from keras.utils import to_categorical

(train_data, train_labels), (test_data, test_labels) = reuters.load_data(
num_words=10000)

def vectorize_sequences(sequences, dimension=10000):
    results = np.zeros((len(sequences), dimension))
    for i, sequence in enumerate(sequences):
        results[i, sequence] = 1
    return results

x_train = vectorize_sequences(train_data)
x_test = vectorize_sequences(test_data)

one_hot_train_labels = to_categorical(train_labels)
one_hot_test_labels = to_categorical(test_labels)

x_val = x_train[:1000]
partial_x_train = x_train[1000:]

y_val = one_hot_train_labels[:1000]
partial_y_train = one_hot_train_labels[1000:]

model = models.Sequential()
model.add(layers.Dense(64, activation='relu', input_shape=(10000,)))
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(46, activation='softmax'))

model.compile(optimizer='rmsprop',
loss='categorical_crossentropy',
metrics=['accuracy'])

history = model.fit(partial_x_train,
                    partial_y_train,
                    epochs=9,
                    batch_size=512,
                    validation_data=(x_val, y_val))

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
results = model.evaluate(x_test, one_hot_test_labels)
print(results)

predictions = model.predict(x_test)
print(predictions[0].shape) # (46,)
print(np.sum(predictions[0])) # 1.0
print(np.argmax(predictions[0])) 
