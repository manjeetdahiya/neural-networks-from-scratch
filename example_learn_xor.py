from neural_network import *
from gradient_descent import *

def load_xor_dataset():
    x_train = []
    y_train = []
    for input, output in zip([(0, 0), (0, 1), (1, 0), (1, 1)], [0, 1, 1, 0]):
        x_train.append(np.array(input).reshape(-1, 1))
        y_train.append(np.array(output).reshape(1, 1))
    return (x_train, y_train), ()

def learn_xor():
    nn = NN([DenseLayer(2, Activation.sigmoid, input_dim=2),
            DenseLayer(1, Activation.linear)])
    nn.set_loss(Loss.cross_entropy_sigmoid)
    nn.init_random_params(seed=10)

    (x_train, y_train), _ = load_xor_dataset()

    opt = StochasticGD(nn, x_train, y_train, x_train, y_train)
    opt.optimize(step_size=0.1, max_epoch=1000, compute_accuracy=Accuracy.bin_class)
    
    # print probabilities
    for input in x_train:
        input = np.array(input).reshape(-1, 1)
        print(f'{input.flatten()} -- {activations.sigmoid(nn.forward_prop_output(input))}')

if __name__ == "__main__":
    learn_xor()
