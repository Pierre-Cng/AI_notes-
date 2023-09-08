# AI_notes
Notes about AI / Machine learning / Deep learning.

## Notes about "Deep Learning with Python" of François Chollet:

Link of the book [here](https://docs.google.com/viewer?a=v&pid=sites&srcid=dW10LmVkdS5wa3xzbmxwfGd4Ojc1ODc1ODY2OTZiOTUzOGQ).  

Core infos:

### Chapter I: What is deep learning?

* distinction between artificial intelligence, machine learning and deep learning.
* Brief history of machine learning.
* Different machine learning approaches mentionned:
    * Probabilistic modeling,
    * Early neural networks,
    * Kernel methods,
    * Decision trees and random forests,
    * Gradient boosting machines.
* Discussion on the actual context and possible future of the AI.

### Chapter II: Before we begin: the mathematical building blocks of neural networks

#### 1) A first look at a neural network

Check demo [here](demo_1.py).

#### 2) Data representations for neural networks

Check demo [here](demo_2.py).  

See also, real world examples of data tensors: 
* Vector data—2D tensors of shape (samples, features)
* Timeseries data or sequence data—3D tensors of shape (samples, timesteps, features)
* Images—4D tensors of shape (samples, height, width, channels) or (samples, channels, height, width)
* Video—5D tensors of shape (samples, frames, height, width, channels) or (samples, frames, channels, height, width)

#### 3) The gears of neural networks: tensor operations

Notions about tensor operations:
* Element-wise operations
* Broadcasting
* Tensor dot
* Tensor reshaping
* Geometric interpretation of tensor operations
* A geometric interpretation of deep learning

#### 4) The engine of neural networks: gradient-based optimization

Notions of:
* What's a derivative?
* Derivative of a tensor operation: the gradient
* Stochastic gradient descent
* Chaining derivatives: the Backpropagation algorithm
* Applying knowledge on code used on [demo 1](demo_1.py)

#### Chapter summary:

* Learning: finding a combination of weights that minimizes a loss function for a given set of training data samples and their corresponding targets.
* It happens by drawing random batches of data samples and their targets, and computing the gradient of the weights with respect to the loss on the batch. The weights are then moved a bit in the opposite direction from the gradient.
* Neural networks are chains of differentiable tensor operations, using chain rule of derivation to find the gradient function mapping the current weights and current batch of data to a gradient value.
* Loss and optimizers are the two functions you need to define before you begin feeding data into a network.
* The loss is the quantity you’ll attempt to minimize during training, so it should represent a measure of success for the task you’re trying to solve.
* The optimizer specifies the exact way in which the gradient of the loss will be used to update parameters: for instance, it could be the RMSProp optimizer, SGD with momentum, and so on.

### Chapter III: Getting started with neural networks

#### 1) Anatomy of a neural network

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

**Introduction to Keras**

![interest chart](https://sites.google.com/view/pierrecongiuconsulting/draft/first-step-in-deep-learning)



