import numpy as np
import matplotlib.pyplot as plt

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
        Generate polynomial features including bias term (column of all 1s).
        
        For example: 
        X = [2, 3], degree = 3
        Output =
        [[1, 2, 4, 8],
        [1, 3, 9, 27]]
        """

        # Convert 1-D input (like [2,3,4]) into 2-D column shape (like [[2],[3],[4]])
        if X.ndim == 1:
            X = X.reshape(-1, 1)

        n_samples = X.shape[0]      # number of rows
        n_features = self.degree + 1  # number of polynomial columns (bias + powers)

        # Create empty feature matrix filled with zeros
        poly = np.zeros((n_samples, n_features))

        # First column = 1s for bias term
        poly[:, 0] = 1

        # Fill remaining columns:
        # poly[:, 1] = X^1
        # poly[:, 2] = X^2
        # ...
        # poly[:, degree] = X^degree
        for d in range(1, self.degree + 1):
            poly[:, d] = X[:, 0] ** d

        return poly

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
    def fit(self, X, y, verbose=True,store_every=100):
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
        self.cost_history_every100=[]

        for _ in range(self.epochs): 
            predictions = X_poly @ self.theta
            errors = predictions - y

            gradients = (1 / m) * (X_poly.T @ errors)
            self.theta -= self.lr * gradients

            cost = (1 / (2 * m)) * np.sum(errors ** 2)
            self.cost_history.append(cost)

            if _ % store_every==0 or _==self.epochs-1:
                self.cost_history_every100.append(cost)
                if verbose:
                    print(f"Epoch {_}: Cost = {cost}")

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
 

    def plot_data_points(self,X,y,y_pred,Title="Data visualization"):
       
        plt.scatter(X,y,color='red',label='Data Points')
        plt.plot(X,y_pred,color='blue')
        plt.title(Title)
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.show()

    def plot_cost_curve(self, title="Training Cost Curve"):
        """
        Plot the cost curve over epochs.
        """
        plt.plot(range(len(self.cost_history)), self.cost_history, color='blue')
        plt.title(title)
        plt.xlabel('Epochs')
        plt.ylabel('Cost')
        plt.show()
    
