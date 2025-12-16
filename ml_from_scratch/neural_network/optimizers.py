import numpy as np
class Optimizer:
    def __init__(self, method="sgd", lr=0.001):
        self.method = method.lower()
        self.lr = lr
        self.v = {}
        self.s = {}
        self.t = 1

    def register(self, name, param):
        self.v[name] = np.zeros_like(param)
        self.s[name] = np.zeros_like(param)

    def update(self, name, param, grad):
        if self.method == "sgd":
            return param - self.lr * grad

        elif self.method == "rmsprop":
            self.s[name] = 0.9 * self.s[name] + 0.1 * (grad ** 2)
            return param - self.lr * grad / (np.sqrt(self.s[name]) + 1e-8)

        elif self.method == "adam":
            self.v[name] = 0.9 * self.v[name] + 0.1 * grad
            self.s[name] = 0.999 * self.s[name] + 0.001 * (grad ** 2)

            v_corr = self.v[name] / (1 - 0.9 ** self.t)
            s_corr = self.s[name] / (1 - 0.999 ** self.t)

            return param - self.lr * v_corr / (np.sqrt(s_corr) + 1e-8)

        else:
            raise ValueError("Unknown optimizer")

    def step(self):
        self.t += 1
