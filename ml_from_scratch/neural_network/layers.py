# neural_network/layers.py
import numpy as np

class DenseLayer:
    def __init__(self, in_dim, out_dim, initializer, activation):
        self.W = initializer(in_dim, out_dim)
        self.b = np.zeros((out_dim, 1))

        self.activation = activation

        # cache
        self.Z = None
        self.A_prev = None
        self.dW = None
        self.db = None

    def forward(self, A_prev):
        self.A_prev = A_prev
        self.Z = self.W @ A_prev + self.b
        return self.activation.forward(self.Z)

    def backward(self, dA):
        dZ = dA * self.activation.backward(self.Z)
        m = self.A_prev.shape[1]

        self.dW = (1 / m) * (dZ @ self.A_prev.T)
        self.db = (1 / m) * np.sum(dZ, axis=1, keepdims=True)
        dA_prev = self.W.T @ dZ

        return dA_prev
