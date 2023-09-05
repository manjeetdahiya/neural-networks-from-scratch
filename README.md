An academic NumPy-based implementation of neural networks from scratch.
The library is capable of defining and training feed-forward neural networks and has been tested on datasets such as MNIST.
It includes a stochastic gradient descent optimizer, accuracy metrics, and a number of activation and loss functions. 
Two examples of using the library are also included.

Following is an example of of defining and training a two-layer network:
```Python
# first layer with input dim of 784
# output layer with 10 neurons for 10 output classes

nn = NN([DenseLayer(16, Activation.relu, input_dim=784),   
         DenseLayer(10, Activation.linear)])

nn.set_loss(Loss.cross_entropy_softmax)
nn.init_random_params()

opt = StochasticGD(nn, x_train, y_train, x_test, y_test)
opt.optimize(step_size=0.01, max_epoch=100, compute_accuracy=Accuracy.multi_class)
```
