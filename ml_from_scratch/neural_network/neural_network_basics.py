import numpy as np
from .initializers import Initializer
from .optimizers import Optimizer
from .activations import Activation


class NeuralNetworkScratch:
    def __init__(self, input_dim=784, hidden_dim=64, output_dim=10,
                 init="he", optimizer="adam", lr=0.001):

        if init == "he":
            self.W1 = Initializer.he(input_dim, hidden_dim)
        elif init == "xavier":
            self.W1 = Initializer.xavier(input_dim, hidden_dim)
        else:
            self.W1 = Initializer.xavier_uniform(input_dim, hidden_dim)

        self.b1 = np.zeros((hidden_dim, 1))
        self.W2 = Initializer.he(hidden_dim, output_dim)
        self.b2 = np.zeros((output_dim, 1))

        self.opt = Optimizer(method=optimizer, lr=lr)
        self.opt.register("W1", self.W1)
        self.opt.register("b1", self.b1)
        self.opt.register("W2", self.W2)
        self.opt.register("b2", self.b2)

    def forward(self, X):
        self.Z1 = self.W1 @ X + self.b1
        self.A1 = Activation.relu(self.Z1)
        self.Z2 = self.W2 @ self.A1 + self.b2
        self.A2 = Activation.softmax(self.Z2)
        return self.A2
    
        def one_hot(self, y):
        y = y.astype(int)
        one_hot_y = np.zeros((10, y.size))
        one_hot_y[y, np.arange(y.size)] = 1
        return one_hot_y

    def backward(self, X, y):
        m = y.size
        y_one_hot = self.one_hot(y)

        dZ2 = self.A2 - y_one_hot
        dW2 = (1 / m) * dZ2 @ self.A1.T
        db2 = (1 / m) * np.sum(dZ2, axis=1, keepdims=True)

        dZ1 = self.W2.T @ dZ2 * Activation.relu_derivative(self.Z1)
        dW1 = (1 / m) * dZ1 @ X.T
        db1 = (1 / m) * np.sum(dZ1, axis=1, keepdims=True)

        return dW1, db1, dW2, db2

    def fit(self, X, y, epochs=100, batch_size=64):
        m = X.shape[1]

        for epoch in range(epochs):
            perm = np.random.permutation(m)
            X = X[:, perm]
            y = y[perm]

            for i in range(0, m, batch_size):
                X_batch = X[:, i:i+batch_size]
                y_batch = y[i:i+batch_size]

                self.forward(X_batch)
                dW1, db1, dW2, db2 = self.backward(X_batch, y_batch)

                self.W1 = self.opt.update("W1", self.W1, dW1)
                self.b1 = self.opt.update("b1", self.b1, db1)
                self.W2 = self.opt.update("W2", self.W2, dW2)
                self.b2 = self.opt.update("b2", self.b2, db2)

                self.opt.step()

    def predict(self, X):
        probs = self.forward(X)
        return np.argmax(probs, axis=0)


