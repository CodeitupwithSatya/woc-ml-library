import numpy as np
import matplotlib.pyplot as plt

class LogisticRegressionFromScratch:
    """
    Binary Logistic Regression using Gradient Descent (from scratch)
    """

    def __init__(self, lr=0.01, n_iters=1000, reg_lambda=1e-4):
        self.lr = lr
        self.n_iters = n_iters
        self.reg_lambda = reg_lambda
        self.w = None
        self.b = None
        self.loss_history = []

    @staticmethod
    def sigmoid(z):
        z = np.clip(z, -500, 500)  # numerical stability...forces the value to be in a reasonable range
        return 1 / (1 + np.exp(-z))

    def _compute_loss(self, y, y_pred):
        """
        Binary Cross Entropy Loss with L2 regularization
        """
        m = len(y)
        eps = 1e-9
        y_pred = np.clip(y_pred, eps, 1 - eps)

        loss = (
            -1 / m * np.sum(
                y * np.log(y_pred) + (1 - y) * np.log(1 - y_pred)
            )
        )

        # L2 regularization (exclude bias)
        reg_loss = (self.reg_lambda / (2 * m)) * np.sum(self.w ** 2)

        return loss + reg_loss

    def fit(self, X, y, store_every=100, verbose=False):

        m, n = X.shape
        y = y.reshape(-1, 1)

        self.w = np.zeros((n, 1))
        self.b = 0.0

        self.loss_history = []        # stores all losses
        self.loss_every_100 = []      # stores checkpoint losses

        for epoch in range(self.n_iters):

            logits = X @ self.w + self.b
            y_pred = self.sigmoid(logits)

            error = y_pred - y

            # Gradients
            dj_dw = (X.T @ error) / m
            dj_dw += (self.reg_lambda / m) * self.w
            dj_db = np.mean(error)

            # Update parameters
            self.w -= self.lr * dj_dw
            self.b -= self.lr * dj_db

            # Compute current loss
            loss = self._compute_loss(y, y_pred)

            # store every loss for graph
            self.loss_history.append(loss)

            # store only checkpoint losses
            if epoch % store_every == 0 or epoch == self.n_iters - 1:
                self.loss_every_100.append(loss)

                if verbose:
                    print(f"Epoch {epoch} | Loss = {loss:.5f}")

        return self


    def predict_proba(self, X):
        """
        Return probability estimates
        """
        logits = X @ self.w + self.b
        return self.sigmoid(logits)

    def predict(self, X, threshold=0.5):
        """
        Return class labels
        """
        probs = self.predict_proba(X)
        return (probs >= threshold).astype(int)
    
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

