import numpy as np

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

    def fit(self, X, y):
        """
        Train logistic regression model
        """
        m, n = X.shape
        y = y.reshape(-1, 1)

        self.w = np.zeros((n, 1))
        self.b = 0.0
        self.loss_history = []

        for _ in range(self.n_iters):
            logits = X @ self.w + self.b
            y_pred = self.sigmoid(logits)

            error = y_pred - y

            # Gradients
            dj_dw = (X.T @ error) / m
            dj_dw += (self.reg_lambda / m) * self.w  # L2 on weights only
            dj_db = np.mean(error)

            # Update
            self.w -= self.lr * dj_dw
            self.b -= self.lr * dj_db

            # Track loss
            loss = self._compute_loss(y, y_pred)
            self.loss_history.append(loss)

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
