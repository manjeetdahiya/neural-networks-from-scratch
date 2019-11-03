import numpy as np
from collections import defaultdict
import random


class NN:
    def __init__(self, layers=None):
        self.layers = []
        self.loss_fun = None
        self._layer_count = 0
        self.params = defaultdict(dict)
        for layer in layers:
            self.add_layer(layer)

    def init_random_params(self, seed=None):
        if seed:
            np.random.seed(seed)
        for layer in self.layers:
            layer.init_random_params()

    def add_layer(self, layer):
        layer._index = self._layer_count
        self._layer_count += 1
        if layer.num_units_input == None:
            layer.num_units_input = self.layers[-1].num_units
        self.layers.append(layer)

    def set_loss(self, loss):
        self.loss_fun = loss

    def forward_prop_output(self, input):
        fwd = self.forward_prop(input)
        return fwd[self.layers[-1]._index]['a']

    def forward_prop_loss(self, Xs, Ys):
        fwd = self.forward_prop(Xs)
        return self.loss_fun(fwd[self.layers[-1]._index]['a'], Ys)

    def forward_prop(self, input):
        forward_prop_output = defaultdict(dict)
        a = input
        for layer in self.layers:
            z, a = layer.forward_prop(a)
            forward_prop_output[layer._index]['z'] = z
            forward_prop_output[layer._index]['a'] = a
        return forward_prop_output

    def back_prop(self, forward_prop_output, x, y):
        grads = defaultdict(dict)
        for i in reversed(range(len(self.layers))):
            layer_l = self.layers[i]
            layer_l_plus_1 = self.layers[i+1] if i+1 < len(self.layers) else None
            layer_l_minus_1 = self.layers[i-1] if i-1 >= 0 else None
            
            is_last_layer = (layer_l_plus_1 is None)
            is_first_layer = (layer_l_minus_1 is None)
            
            z = forward_prop_output[layer_l._index]['z']
            a = forward_prop_output[layer_l._index]['a']
            
            if is_last_layer:
                d_a_z = layer_l.activation_func(z, derivative=True)
                da = self.loss_fun(a, y, derivative=True)
                dz = d_a_z * da
            else:
                dz_l_plus_1 = grads[layer_l_plus_1._index]['dz']
                dz = layer_l.back_prop_dz(z, dz_l_plus_1, layer_l_plus_1)
            grads[layer_l._index]['dz'] = dz
            
            if is_first_layer:
                a_l_minus_1 = x
            else:
                a_l_minus_1 = forward_prop_output[layer_l_minus_1._index]['a']
            
            db, dW = self.back_prop_compute_derivative_wrt_params(a_l_minus_1, dz)
            grads[layer_l._index]['db'] = db
            grads[layer_l._index]['dW'] = dW
            
            assert dW.shape == layer_l.get_param_W().shape
            assert db.shape == layer_l.get_param_b().shape
        return grads

    def back_prop_compute_derivative_wrt_params(self, a_prev, dz):
        db = dz  # (units_l x 1)
        dW = np.matmul(a_prev, dz.T)  # (units_l-1 x 1) x (units_l x 1)'
        return (db, dW)

    def iter_params(self):
        params = self.get_params()
        for layer_idx in params:
            for wt_name in params[layer_idx]:
                for idx in np.ndindex(params[layer_idx][wt_name].shape):
                    yield (layer_idx, wt_name, idx)

    def get_params(self):
        params = defaultdict(dict)
        for layer in self.layers:
            params[layer._index]['W'] = layer.get_param_W()
            params[layer._index]['b'] = layer.get_param_b()
        return params

    def set_params(self, params):
        for layer in self.layers:
            W = params[layer._index]['W']
            b = params[layer._index]['b']
            layer.set_params(W, b)

    def set_param(self, layer, name, idx, val):
        self.layers[layer].params_layer[name][idx] = val

    def get_param(self, layer, name, idx):
        return self.layers[layer].params_layer[name][idx]

    def get_grads(self, Xs, Ys):
        fwd = self.forward_prop(Xs)
        grads = self.back_prop(fwd, Xs, Ys)
        return grads

    def grad_check(self, Xs, Ys):
        grads = self.back_prop(self.forward_prop(Xs), Xs, Ys)
        delta = 0.00000001
        
        for wt_idx in self.iter_params():
            layer_idx, wt_name, np_idx = wt_idx
            y_old = self.forward_prop_loss(Xs, Ys)
            wt_old = self.get_param(*wt_idx)
            wt_new = wt_old + delta
            self.set_param(*wt_idx, wt_new)
            y_new = self.forward_prop_loss(Xs, Ys)
            A = (y_new - y_old)/delta
            B = grads[layer_idx]["d"+wt_name][np_idx]
            if abs(A - B) > 0.00001:
                print(f'grad check failed: {A}, {B}, {A-B}')
            
            # reset weight 
            self.set_param(*wt_idx, wt_old)


class DenseLayer:
    def __init__(self, num_units, activation, input_dim=None):
        self.num_units = num_units
        self.activation_func = activation
        self._index = None
        self.num_units_input = input_dim
        self.params_layer = {}

    def init_random_params(self):
        W = np.random.randn(*self.get_W_shape())
        b = np.random.randn(*self.get_b_shape())
        self.set_params(W, b)

    def set_params(self, W, b):
        assert W.shape == self.get_W_shape()
        assert b.shape == self.get_b_shape()
        self.params_layer['W'] = W
        self.params_layer['b'] = b

    def get_param_W(self):
        return self.params_layer['W']

    def get_param_b(self):
        return self.params_layer['b']

    def get_W_shape(self):
        return (self.num_units_input, self.num_units)

    def get_b_shape(self):
        return (self.num_units, 1)

    def input_shape(self):
        return (self.num_units_input, 1)

    def output_shape(self):
        return (self.num_units, 1)

    def forward_prop(self, input):
        assert input.shape == self.input_shape()
        z = np.matmul(self.get_param_W().T, input) + self.get_param_b()
        a = self.activation_func(z)
        assert a.shape == self.output_shape()
        return (z, a)

    def back_prop_dz(self, z, dz_next, layer_next):
        assert dz_next.shape == layer_next.output_shape()
        a = np.matmul(layer_next.get_param_W(), dz_next)
        b = self.activation_func(z, derivative=True)
        return np.multiply(b, a)

class Loss:
    @staticmethod
    def cross_entropy(a_last, y, derivative=False):
        if not derivative:
            ce = -y * np.log(a_last) - (1 - y) * np.log(1 - a_last)
            return np.sum(ce)
        else:
            da_last = -y/a_last + (1 - y)/(1 - a_last)
            return da_last

    @staticmethod
    def cross_entropy_sigmoid(a_last, y, derivative=False):
        if not derivative:
            'a_last is logit'
            a_last_sm = activations.sigmoid(a_last)
            ce = -y * np.log(a_last_sm) - (1 - y) * np.log(1 - a_last_sm)
            assert not np.isnan(ce)
            return np.sum(ce)
        else:
            a_last_sm = activations.sigmoid(a_last)
            der = a_last_sm - y
            return der

    @staticmethod
    def cross_entropy_softmax(a_last, y, derivative=False):
        if not derivative:
            'a_last is logit'
            a_last_sm = Loss.softmax(a_last)
            ce = -y * np.log(a_last_sm)
            return np.sum(ce)
        else:
            a_last_sm = Loss.softmax(a_last)
            der = a_last_sm - y
            return der

    @staticmethod
    def softmax(X):
        Xrel = X - np.max(X)  # to avoid potential overflow, results remain the same though
        a = np.exp(Xrel)
        return a/np.sum(a)

    @staticmethod
    def linear(a_last, y, derivative=False):
        if not derivative:
            ce = a_last - y
            return np.sum(ce)
        else:
            return 1


class activations:
    @staticmethod
    def sigmoid(X, derivative=False):
        if not derivative:
            return 1/(1 + np.exp(-X))
        else:
            s = 1/(1 + np.exp(-X))
            return s * (1 - s)

    @staticmethod
    def relu(X, derivative=False):
        if not derivative:
            return np.maximum(X, 0)
        else:
            return (X > 0).astype(int)

    @staticmethod
    def linear(X, derivative=False):
        if not derivative:
            return X
        else:
            return 1


class accuracy:
    @staticmethod
    def multi_class(nn, Xs, Ys):
        preds = 0
        for x, y in zip(Xs, Ys):
            fwd = nn.forward_prop(x)
            a = fwd[len(fwd)-1]["a"]
            preds += int(np.argmax(a) == np.argmax(y))
        return preds*100/len(Xs)

    @staticmethod
    def bin_class(nn, Xs, Ys):
        preds = 0
        for x, y in zip(Xs, Ys):
            fwd = nn.forward_prop(x)
            a = fwd[len(fwd)-1]["a"]
            a = [[1]] if a >= 0.5 else [[0]]
            preds += int(a == y)
        return preds*100/len(Xs)
