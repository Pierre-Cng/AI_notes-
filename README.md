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
