from scipy.io import loadmat
import os
import numpy as np

def np_sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def np_tanh(x):
    return np.tanh(x)


def np_relu(x):
    return np.maximum(0, x)

class NeuralNet(object):
    def __init__(self, x_train, y_train):
        self.x_train = x_train
        self.y_train = y_train
        self.numberof_input = self.x_train.shape[0]
        self.numberof_output = self.y_train.shape[0]
        self.task = np.array([[self.numberof_input, 7, self.numberof_output],
                              [self.numberof_input, 8, self.numberof_output],
                              [self.numberof_input, 9, self.numberof_output]
            ])
        self.numberof_layers = max(len(self.task[i]) - 1 for i in range(len(self.task)))
        self.task_max = [max(self.task[:, layer]) for layer in range(self.numberof_layers + 1)]
        self.pop_size = self.init_pop()
        self.functions = [self.forward_eval] * len(self.task)
        print (self.task)

    def init_pop(self):
        L = len(self.task_max)
        pop_size = 0
        for layer in range(1, L):
            pop_size += self.task_max[layer] * self.task_max[layer-1] + self.task_max[layer]

        return pop_size

    def forward_eval(self, individual, skill_factor, is_eval_acc=False):
        L = len(self.task[skill_factor]) - 1 # L = 2
        layers = self.task[skill_factor]

        A = self.x_train
        Y = self.y_train
        start = 0

        for l in range(1, L):
            Wl = individual[start:start + layers[l] * layers[l - 1]].reshape(layers[l], layers[l - 1]) # 0 -> 7
            start += self.task_max[l] * self.task_max[l-1] # 9
            bl = individual[start:start + layers[l]].reshape(layers[l], 1) # 7-> 16
            start = self.task_max[l] * self.task_max[l-1] + self.task_max[l] # 9 + 9 = 18

            Z = np.dot(Wl, A) + bl
            A = np_relu(Z)

        WL = individual[start:start + layers[L] * layers[L - 1]].reshape(layers[L], layers[L - 1])
        start += self.task_max[L] * self.task_max[L-1]
        bL = individual[start:start + layers[L]].reshape(layers[L], 1)

        ZL = np.dot(WL, A) + bL
        AL = np_sigmoid(ZL)
        cost = 0.5 * np.mean((Y - AL) ** 2)

        lambd = 1.0
        m = Y.shape[1]
        L2_regularization_cost = 0

        start = 0
        for l in range(1, L + 1):
            # L2_regularization_cost = L2_regularization_cost + np.sum(np.square(self.parameters['W' + str(l)]))
            L2_regularization_cost = L2_regularization_cost + np.sum(np.square(individual[start:layers[l] * layers[l - 1]]))
            start = self.task_max[l] * self.task_max[l-1] + self.task_max[l]

        L2_regularization_cost = lambd / (2 * m) * L2_regularization_cost

        factorial_costs = cost + L2_regularization_cost

        # if is_eval_acc:
        #     acc = ((Y > 0.5) == (AL > 0.5))
        #     self.accuracy[self.skill_factor] = np.sum(acc) / len(acc.squeeze())

        return factorial_costs



