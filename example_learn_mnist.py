from neural_network import *
from gradient_descent import *

def learn_mnist():
    nn = NN([DenseLayer(16, activations.relu, input_dim=784),
            DenseLayer(10, activations.linear)])
    nn.set_loss(Loss.cross_entropy_softmax)
    nn.init_random_params()

    (x_train, y_train), (x_test, y_test) = load_mnist_dataset()

    opt = StochasticGD(nn, x_train, y_train, x_test, y_test)
    opt.optimize(step_size=0.01, max_epoch=100, compute_accuracy=accuracy.multi_class)


def load_mnist_dataset():
    import tensorflow as tf
    mnist = tf.keras.datasets.mnist
    from keras.utils import to_categorical
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    
    y_train = to_categorical(y_train)
    y_test = to_categorical(y_test)
    y_train = y_train.reshape((-1, 10, 1))
    y_test = y_test.reshape((-1, 10, 1))
    
    x_train = x_train.reshape((-1, 784, 1))/255
    x_test = x_test.reshape((-1, 784, 1))/255
    
    return (x_train, y_train), (x_test, y_test)


if __name__ == "__main__":
    learn_mnist()
