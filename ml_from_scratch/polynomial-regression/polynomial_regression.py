import numpy as np

class PolynomialRegression:
    """
    Polynomial Regression implemented from scratch using Gradient Descent.
    """

    def __init__(self, degree=2, lr=0.01, epochs=1000, normalize=True):
        """
        Parameters
        ----------
        degree : int
            Degree of polynomial
        lr : float
            Learning rate
        epochs : int
            Number of gradient descent iterations
        normalize : bool
            Whether to apply z-score normalization
        """
        self.degree = degree
        self.lr = lr
        self.epochs = epochs
        self.normalize = normalize

        self.theta = None
        self.cost_history = []
        self.mean = None
        self.std = None

    # -------------------------
    # Feature Engineering
    # -------------------------
    def _polynomial_features(self, X):
        """
        Generate polynomial features with bias term.
        """
        X = X.reshape(-1, 1) if X.ndim == 1 else X
        features = [np.ones((X.shape[0], 1))]

        for d in range(1, self.degree + 1):
            features.append(X ** d)

        return np.hstack(features)

    # -------------------------
    # Normalization (from your code)
    # -------------------------
    def _normalize(self, X):
        self.mean = np.mean(X, axis=0)
        self.std = np.std(X, axis=0) + 1e-8
        return (X - self.mean) / self.std

    # -------------------------
    # Training
    # -------------------------
    def fit(self, X, y):
        """
        Train the model using Gradient Descent.
        """
        y = y.reshape(-1, 1)

        X_poly = self._polynomial_features(X)

        if self.normalize:
            X_poly[:, 1:] = self._normalize(X_poly[:, 1:])

        m, n = X_poly.shape
        self.theta = np.zeros((n, 1))
        self.cost_history = []

        for _ in range(self.epochs):
            predictions = X_poly @ self.theta
            errors = predictions - y

            gradients = (1 / m) * (X_poly.T @ errors)
            self.theta -= self.lr * gradients

            cost = (1 / (2 * m)) * np.sum(errors ** 2)
            self.cost_history.append(cost)

        return self

    # -------------------------
    # Prediction
    # -------------------------
    def predict(self, X):
        """
        Predict target values.
        """
        X_poly = self._polynomial_features(X)

        if self.normalize:
            X_poly[:, 1:] = (X_poly[:, 1:] - self.mean) / self.std

        return X_poly @ self.theta

    # -------------------------
    # Evaluation
    # -------------------------
    def score(self, X, y):
        """
        Compute R² score.
        """
        y = y.flatten()
        y_pred = self.predict(X).flatten()

        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)

        return 1 - ss_res / ss_tot if ss_tot != 0 else 0.0
