import numpy as np
from collections import Counter

class KNNFromScratch:
    """
    K-Nearest Neighbors (KNN) classifier implemented from scratch
    """

    def __init__(self, k=3):
        self.k = k
        self.X_train = None
        self.y_train = None

    @staticmethod
    def euclidean_distance(x1, x2):
        return np.sqrt(np.sum((x1 - x2) ** 2))

    def fit(self, X, y):
        """
        Store training data (lazy learner)
        """
        self.X_train = X
        self.y_train = y

    def _predict_single(self, x):
        """
        Predict label for a single sample
        """
        # Compute distances
        distances = [
            self.euclidean_distance(x, x_train)
            for x_train in self.X_train
        ]

        # Get k nearest neighbors
        k_indices = np.argsort(distances)[:self.k]
        k_labels = [self.y_train[i] for i in k_indices]

        # Majority vote
        most_common = Counter(k_labels).most_common(1)
        return most_common[0][0]

    def predict(self, X):
        """
        Predict class labels for samples in X
        """
        return np.array([self._predict_single(x) for x in X])

    def predict_proba(self, X):
        """
        Return class probabilities (for ROC / analysis)
        """
        probs = []

        for x in X:
            distances = [
                self.euclidean_distance(x, x_train)
                for x_train in self.X_train
            ]
            k_indices = np.argsort(distances)[:self.k]
            k_labels = [self.y_train[i] for i in k_indices]

            counts = Counter(k_labels)
            total = sum(counts.values())

            class_probs = {cls: count / total for cls, count in counts.items()}
            probs.append(class_probs)

        return probs
