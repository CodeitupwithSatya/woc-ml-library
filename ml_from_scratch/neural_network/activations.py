# neural_network/activations.py
import numpy as np

class ReLU:
    def forward(self, Z):
        return np.maximum(0, Z)

    def backward(self, Z):
        return (Z > 0).astype(float)


class Softmax:
    def forward(self, Z):
        expZ = np.exp(Z - np.max(Z, axis=0, keepdims=True))
        return expZ / np.sum(expZ, axis=0, keepdims=True)

    def backward(self, Z):
        # handled with cross-entropy
        return np.ones_like(Z)

class Sigmoid:
    def forward(self, Z):
        return 1 / (1 + np.exp(-Z))

    def backward(self, Z):
        return self.forward(Z) * (1 - self.forward(Z))