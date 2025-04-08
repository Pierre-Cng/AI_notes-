# Notes about "Deep Learning with Python" of François Chollet:

Link of the book [here](https://docs.google.com/viewer?a=v&pid=sites&srcid=dW10LmVkdS5wa3xzbmxwfGd4Ojc1ODc1ODY2OTZiOTUzOGQ).  

Core infos:

## Chapter I: What is deep learning?

* distinction between artificial intelligence, machine learning and deep learning.
* Brief history of machine learning.
* Different machine learning approaches mentionned:
    * Probabilistic modeling,
    * Early neural networks,
    * Kernel methods,
    * Decision trees and random forests,
    * Gradient boosting machines.
* Discussion on the actual context and possible future of the AI.

## Chapter II: Before we begin: the mathematical building blocks of neural networks

### 1) A first look at a neural network

Check demo [here](demo_1.py).

### 2) Data representations for neural networks

Check demo [here](demo_2.py).  

See also, real world examples of data tensors: 
* Vector data—2D tensors of shape (samples, features)
* Timeseries data or sequence data—3D tensors of shape (samples, timesteps, features)
* Images—4D tensors of shape (samples, height, width, channels) or (samples, channels, height, width)
* Video—5D tensors of shape (samples, frames, height, width, channels) or (samples, frames, channels, height, width)

### 3) The gears of neural networks: tensor operations

Notions about tensor operations:
* Element-wise operations
* Broadcasting
* Tensor dot
* Tensor reshaping
* Geometric interpretation of tensor operations
* A geometric interpretation of deep learning

### 4) The engine of neural networks: gradient-based optimization

Notions of:
* What's a derivative?
* Derivative of a tensor operation: the gradient
* Stochastic gradient descent
* Chaining derivatives: the Backpropagation algorithm
* Applying knowledge on code used on [demo 1](demo_1.py)

### Chapter summary:

* Learning: finding a combination of weights that minimizes a loss function for a given set of training data samples and their corresponding targets.
* It happens by drawing random batches of data samples and their targets, and computing the gradient of the weights with respect to the loss on the batch. The weights are then moved a bit in the opposite direction from the gradient.
* Neural networks are chains of differentiable tensor operations, using chain rule of derivation to find the gradient function mapping the current weights and current batch of data to a gradient value.
* Loss and optimizers are the two functions you need to define before you begin feeding data into a network.
* The loss is the quantity you’ll attempt to minimize during training, so it should represent a measure of success for the task you’re trying to solve.
* The optimizer specifies the exact way in which the gradient of the loss will be used to update parameters: for instance, it could be the RMSProp optimizer, SGD with momentum, and so on.

## Chapter III: Getting started with neural networks

### 1) Anatomy of a neural network

**Layers: the building blocks of deep learning**  

* Simple vector data, stored in 2D tensors of shape (samples, features) -> Dense class in Keras
* Sequence data, stored in 3D tensors of shape (samples, timesteps, features) -> recurrent layers such as an LSTM layer.
* Image data, stored in 4D tensors -> 2D convolution layers (Conv2D)

**Models: networks of layers**  

More than linear stack of layers, mapping a single input to a single output, variety of network topologies exists such as:
* Two-branch networks
* Multihead networks
* Inception blocks

> "Picking the right network architecture is more an art than a science; and although there are some best practices and principles you can rely on, only practice can help you become a proper neural-network architect."

**Loss functions and optimizers: keys to configuring the learning process**

Use:
* Binary crossentropy for a two-class classification problem,
* Categorical crossentropy for a many-class classification problem,
* Meansquared error for a regression problem,
* Connectionist temporal classification (CTC) for a sequence-learning problem.

### 2) Introduction to Keras

[interest chart](https://trends.google.com/trends/explore?cat=1299&date=all&q=Keras,TensorFlow,Theano,Torch,Pytorch&hl=fr)

**Keras, TensorFlow, Theano, and CNTK**
 
Three existing backend implementations: 
* TensorFlow
* Theano
* Microsoft Cognitive Toolkit (CNTK)

Keras is able to run on CPUs and GPUs:
* On CPU -> using Eigen (http://eigen.tuxfamily.org)
* On GPU -> using NVIDIA CUDA Deep Neural Network library (cuDNN).

**Developing with Keras: a quick overview**

Keras workflow looks just like that example:
1) Define your training data: input tensors and target tensors.
2) Define a network of layers (or model ) that maps your inputs to your targets.
3) Configure the learning process by choosing a loss function, an optimizer, and some metrics to monitor.
4) Iterate on your training data by calling the fit() method of your model.

Two ways to define a model: 
* Using the Sequential class (only for linear stacks of layers)
* using the functional API (for directed acyclic graphs of layers, which lets you build completely arbitrary architectures).

### 3) Setting up a deep-learning workstation

Run Keras experiment on a Jupiter notebook (recommended) local or cloud depending on budget and GPU capacity. 

### 4) Classifying movie reviews: a binary classification example

Check demo [here](demo_3.py).  
Check demo [here](demo_3_1.py).  

NB:
* Sequences of words can be encoded as binary vectors for example.
* Stacks of Dense layers with relu activations can solve a wide range of problems (including sentiment classification).
* In a binary classification problem (two output classes) -> network should end with a Dense layer with one unit and a sigmoid activation: the output of the network should be a scalar between 0 and 1, encoding a probability.
* With such a scalar sigmoid output on a binary classification problem -> loss function is binary_crossentropy.
* The rmsprop optimizer is a good enough choice for any problem.
* Be careful of overfitting. Be sure to always monitor performance on data that is outside of the training set.

### 5) Classifying newswires: a multiclass classification example

Check demo [here](demo_4.py).  

Check demo [here](demo_4_1.py).  

With integer labels, you should use sparse_categorical_crossentropy:
```model.compile(optimizer='rmsprop', loss='sparse_categorical_crossentropy', metrics=['acc'])```

NB:
* Classify data points among N classes -> network should end with a Dense layer of size N.
* In a single-label, multiclass classification problem -> network should end with a softmax activation so that it will output a probability distribution over the N output classes.
* Categorical crossentropy is almost always the loss function to use for such problems. It minimizes the distance between the probability distributions output by the network and the true distribution of the targets.
* 2 ways to handle labels in multiclass classification:
    – Encoding the labels via categorical encoding (also known as one-hot encoding) and using categorical_crossentropy as a loss function
    – Encoding the labels as integers and using the sparse_categorical_crossentropy loss function
* When large number of categories -> avoid information bottlenecks due to too small intermediate layers.

### 6) Predicting house prices: a regression example

Check demo [here](demo_5.py).  

Check demo [here](demo_5_1.py).  

NB:
* Regression -> different loss functions than classification -> Mean squared error (MSE) is a common loss function.
* Different metrics for regression -> accuracy doesn’t apply for regression -> common regression metric is mean absolute error (MAE).
* When input data have values in different ranges -> should be scaled independently as a preprocessing step.
* When little data available -> K-fold validation is a great way to reliably evaluate a model.
* When little training data -> use a small network with few hidden layers (typically only one or two) to avoid overfitting.

### Chapter summary:

Notions:
* Tasks on vector data: binary classification, multiclass classification, and scalar regression. 
* Preprocess raw data, scale each feature.
* Avoid overfitting.
* Small training data = small network with only one or two hidden layers, to avoid overfitting.
* Little data = K-fold validation can help evaluate your model.
* /!\ Info bottleneck happened when intermediate layers too small.
* Regression = different loss functions / metrics than classification.

## Chapter IV: Fundamentals of machine learning

### 1) Four branches of machine learning

**Supervised learning**

**Unsupervised learning**

**Self-supervised learning**

**Reinforcement learning** 


machine-learning-specific definitions:
* Sample or input: One data point that goes into your model.
* Prediction or output: What comes out of your model.
* Target: The truth. What your model should ideally have predicted, according to an external source of data.
* Prediction error or loss value: measure of distance between your model’s prediction and the target.
* Classes:  set of possible labels to choose from in a classification problem.
* Label: specific instance of a class annotation in a classification problem.
* Ground-truth or annotations: All targets for a dataset, typically collected by humans.
* Binary classification: each input sample should be categorized into two exclusive categories.
* Multiclass classification: each input sample should be categorized into more than two categories.
* Multilabel classification: each input sample can be assigned multiple labels.
* Scalar regression: target is a continuous scalar value (e.g. Predicting house prices).
* Vector regression: target is a set of continuous values (such as the coordinates of a bounding box in an image).
* Mini-batch or batch: A small set of samples (typically between 8 and 128) processed simultaneously by the model. 

### 2) Evaluating machine-learning models

**Training, validation, and test sets**

NB:
* Splitting  data into three sets: training, validation, and test. 
* The number of layers or the size of the layers is called hyperparameters of the model (!= parameters -> network’s weights). 
* Three classic evaluation recipes to split data: simple hold-out validation, K-fold validation, and iterated K-fold validation with shuffling.
* A good practice: randomly shuffle your data in the first place.
* If you’re trying to predict the future given the past -> DO NOT randomly shuffle your data before splitting it. You should always make sure all data in your test set is posterior to the data in the training set. (You want to test the futur on past trained data)
* If some data points in your data appear twice -> Make sure your training set and validation don't contain the same data.

### 3) Data preprocessing, feature engineering, and feature learning

**Data preprocessing for neural networks**

* Vectorization
* Value normalization
* Handling missing values

### 4) Overfitting and underfitting

**Regularization technics:**

* Reducing the network’s size