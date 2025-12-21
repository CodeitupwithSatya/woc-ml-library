import numpy as np
import matplotlib.pyplot as plt

class LinearRegression:
    """
    Linear Regression implemented from scratch using Gradient Descent.
    Supports optional L2 regularization.
    """

    def __init__(self, lr=0.01, epochs=1000, lambda_reg=0.0):
        self.lr = lr
        self.epochs = epochs
        self.lambda_reg = lambda_reg

        self.weights = None
        self.loss_history = []

    def _initialize_weights(self, n_features):
        self.weights = np.zeros((n_features, 1))

    def _compute_loss(self, X, y):
        m = X.shape[0]
        y_pred = X @ self.weights
        error = y_pred - y

        mse_loss = (1 / (2 * m)) * np.sum(error ** 2)

        reg_loss = (self.lambda_reg / (2 * m)) * np.sum(self.weights[1:] ** 2)
        return mse_loss + reg_loss

    def fit(self, X, y, store_every=100, verbose=False):
        """
        store_every : int
            store loss every N epochs
        """
        X = X.astype(float)
        y = y.astype(float)

        m, n = X.shape
        self._initialize_weights(n)
        
        self.loss_history = []        # stores all losses
        self.loss_every_100 = []     # stores checkpoint losses

        for epoch in range(self.epochs):

            # forward pass
            y_pred = X @ self.weights
            error = y_pred - y

            # gradient computation
            gradient = (1 / m) * (X.T @ error)

            # regularization except bias term
            reg_gradient = (self.lambda_reg / m) * self.weights
            reg_gradient[0] = 0
            gradient += reg_gradient

            # update weights
            self.weights -= self.lr * gradient

            loss=self._compute_loss(X, y)

            self.loss_history.append(loss)

            # compute loss
            
            if epoch % store_every == 0 or epoch == self.epochs - 1:
                self.loss_every_100.append(loss)

                if verbose:
                    print(f"Epoch {epoch} | Loss = {loss:.5f}")

        return self

    def predict(self, X):
        return X @ self.weights

    def plot_loss_curve(self, title="Training Loss Curve"):
        if not isinstance(self.loss_history, (list, tuple)):
            raise ValueError("loss_history must be a list or tuple")

        plt.figure(figsize=(7,5))
        plt.plot(self.loss_history, marker='o')
        plt.xlabel("Checkpoint Epoch")
        plt.ylabel("Loss")
        plt.title(title)
        plt.grid(True)
        plt.show()
