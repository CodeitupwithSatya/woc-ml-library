import numpy as np

class LinearRegression:
    """
    Linear Regression implemented from scratch using Gradient Descent.
    Supports optional L2 regularization.
    """

    def __init__(self, lr=0.01, epochs=1000, lambda_reg=0.0):
        """
        Parameters
        ----------
        lr : float
            Learning rate
        epochs : int
            Number of gradient descent iterations
        lambda_reg : float
            L2 regularization strength (0 = no regularization)
        """
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

    def fit(self, X, y):
        """
        Train the linear regression model.

        Parameters
        ----------
        X : ndarray of shape (m, n)
            Feature matrix (bias term should already be included if needed)
        y : ndarray of shape (m, 1)
            Target values
        """
        X = X.astype(float)
        y = y.astype(float)

        m, n = X.shape
        self._initialize_weights(n)

        for _ in range(self.epochs):
            y_pred = X @ self.weights
            error = y_pred - y

            gradient = (1 / m) * (X.T @ error)

            # L2 Regularization (exclude bias)
            reg_gradient = (self.lambda_reg / m) * self.weights
            reg_gradient[0] = 0
            gradient += reg_gradient

            self.weights -= self.lr * gradient

            loss = self._compute_loss(X, y)
            self.loss_history.append(loss)

        return self

    def predict(self, X):
        """
        Predict target values.

        Parameters
        ----------
        X : ndarray of shape (m, n)

        Returns
        -------
        y_pred : ndarray of shape (m, 1)
        """
        return X @ self.weights

    def score(self, X, y):
        """
        Compute R² score.

        Parameters
        ----------
        X : ndarray
        y : ndarray

        Returns
        -------
        r2 : float
        """
        y = y.flatten()
        y_pred = self.predict(X).flatten()

        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)

        return 1 - ss_res / ss_tot if ss_tot != 0 else 0.0
