import numpy as np
# from collections import defaultdict
# import random

class StochasticGD:
    def __init__(self, nn, x_test, y_test, x_train, y_train):
        self.nn = nn
        self.epoch_print_step = 1
        self.x_test = x_test
        self.y_test = y_test
        self.x_train = x_train
        self.y_train = y_train

    @staticmethod
    def grads_norm(params):
        ps = np.array([])
        for idx in params:
            ps = np.append(ps, params[idx]['dW'].reshape((-1,)))
            ps = np.append(ps, params[idx]['db'].reshape((-1,)))
            # print(f'db {params[idx]["db"]}')
        return np.linalg.norm(ps)

    @staticmethod
    def params_subtract(params, step_size, grads):
        change_W = 0
        change_b = 0
        for idx in params:
            params[idx]['W'] = params[idx]['W'] - step_size * grads[idx]['dW']
            params[idx]['b'] = params[idx]['b'] - step_size * grads[idx]['db']
            # print(np.sum(step_size * np.abs(grads[idx]['db'])))
            if idx == 1:
                change_W += 100 * np.sum(step_size * np.abs(grads[idx]['dW'])/np.abs(params[idx]['W']))
                change_b += 100 * np.sum(step_size * np.abs(grads[idx]['db'])/np.abs(params[idx]['b']))
        # if old != new:
        #     print(f'% change in params: {(old-new)*100/old}')
        return params, change_W, change_b


    def compute_loss(self, Xs, Ys):
        loss = 0
        for x, y in zip(Xs, Ys):
            loss += self.nn.forward_prop_loss(x, y)
        return loss

    def check_metrics(self, epoch, compute_accuracy):
        if epoch % self.epoch_print_step == 0:
            loss = self.compute_loss(self.x_train, self.y_train)
            loss_test = self.compute_loss(self.x_test, self.y_test)
            if compute_accuracy:
                preds = compute_accuracy(self.nn, self.x_train, self.y_train)
                preds_test = compute_accuracy(self.nn, self.x_test, self.y_test)
                print(f'loss: {loss} -- {preds} -||- {loss_test} -- {preds_test}')
            else:
                print(f'loss: {loss} -||- {loss_test}')

    def optimize(self, step_size=0.1, max_epoch=100, compute_accuracy=None):
        N = len(self.x_train)
        for i in range(max_epoch):
            self.check_metrics(i, compute_accuracy)
            change_W = 0
            change_b = 0
            for x, y in zip(self.x_train, self.y_train):
                grads = self.nn.get_grads(x, y)
                params, change_W_, change_b_ = self.params_subtract(self.nn.get_params(), step_size, grads)
                self.nn.set_params(params)
                change_W += change_W_
                change_b += change_b_
            print(f'change_W: {change_W/N} change_b: {change_b/N}')
            # nn.grad_check(Xs[100], Ys[100])
        return params
