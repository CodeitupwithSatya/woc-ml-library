# neural_network/neural_network.py
import numpy as np
from .layers import DenseLayer
from .activations import ReLU, Softmax
from .initializers import Initializer
from .optimizers import Optimizer

class NeuralNetworkScratch:
    def __init__(self, layer_dims, optimizer="sgd", lr=0.01):
        """
        layer_dims example: [128, 64, 10]
        """
        self.layers = []
        self.optimizer = Optimizer(method=optimizer, lr=lr)

        for i in range(len(layer_dims) - 2):
            self.layers.append(
                DenseLayer(
                    layer_dims[i],
                    layer_dims[i + 1],
                    Initializer.he,
                    ReLU()
                )
            )

        # Output layer
        self.layers.append(
            DenseLayer(
                layer_dims[-2],
                layer_dims[-1],
                Initializer.xavier,
                Softmax()
            )
        )

        # register parameters
        for idx, layer in enumerate(self.layers):
            self.optimizer.register(f"W{idx}", layer.W)
            self.optimizer.register(f"b{idx}", layer.b)

    def forward(self, X):
        A = X
        for layer in self.layers:
            A = layer.forward(A)
        return A

    def backward(self, Y, Y_hat):
        m = Y.shape[1]
        dA = Y_hat - Y  # softmax + CE

        for layer in reversed(self.layers):
            dA = layer.backward(dA)

    def update(self):
        for idx, layer in enumerate(self.layers):
            layer.W = self.optimizer.update(f"W{idx}", layer.W, layer.dW)
            layer.b = self.optimizer.update(f"b{idx}", layer.b, layer.db)
        self.optimizer.step()

    def fit(self, X, y, epochs=100):
        Y = self._one_hot(y)

        for epoch in range(epochs):
            Y_hat = self.forward(X)
            self.backward(Y, Y_hat)
            self.update()

            if epoch % 10 == 0:
                loss = -np.mean(np.sum(Y * np.log(Y_hat + 1e-8), axis=0))
                print(f"Epoch {epoch}: Loss = {loss:.4f}")

    def predict(self, X):
        probs = self.forward(X)
        return np.argmax(probs, axis=0)

    def _one_hot(self, y):
        y = y.astype(int)
        one_hot = np.zeros((np.max(y) + 1, y.size))
        one_hot[y, np.arange(y.size)] = 1
        return one_hot
