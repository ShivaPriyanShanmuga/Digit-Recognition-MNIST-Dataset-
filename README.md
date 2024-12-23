# Digit-Recognition-MNIST-Dataset-

**Neural Network for MNIST Digit Recognition**
-------------------------------------------------

This project implements a simple neural network from scratch in Python to classify digits from the MNIST dataset. 

The implementation uses Numpy for numerical operations and Keras for loading the dataset.

**Features**

-> MNIST Dataset Handling: Prepares the MNIST dataset by normalizing and flattening the input images.

-> Custom Neural Network: A three-layer neural network built from scratch using sigmoid activation functions.

-> Forward Propagation: Calculates activations and the cost for a given input and weights.

-> Backward Propagation: Updates weights and biases based on the cost using gradient descent.

-> Gradient Descent Optimization: Trains the network over multiple iterations to minimize the cost.


**Requirements**

->  Python 3.7+

->  Numpy

->  Keras

**The script will:**

. Train the neural network on the MNIST training dataset.

. Evaluate the performance on the MNIST test dataset.

. Print the cost at various stages of training and testing.(After every 1000 iterations)

**Implementation Details**

. **Data Preparation :**

-> The MNIST images are flattened into 1D arrays of 784 elements.

-> Labels are converted to one-hot encoded vectors.

. **Neural Network Architecture**

-> Input Layer: 784 nodes (one for each pixel in the flattened input).

-> Hidden Layers:

      . Layer 1: 16 nodes.
      
      . Layer 2: 16 nodes.
      
-> Output Layer: 10 nodes (one for each digit class).

-> Activation Function: Sigmoid for all layers.

**Training**

The gradient_descent function trains the network.

-> Learning Rate (alpha): Default is 0.9.

-> Iterations: Default is 59999.

**Testing**

The script evaluates the model's performance on 10,000 test samples and computes the average cost.

**Customization**

Learning Rate: Adjust the alpha parameter in the gradient_descent function to modify the learning rate.

Number of Iterations: Modify the iterations parameter in the gradient_descent function to change the training duration.
