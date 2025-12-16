import numpy as np


# =========================
# Decision Tree Node
# =========================
class DecisionTreeNode:
    def __init__(
        self,
        feature_index=None,
        threshold=None,
        is_numeric=True,
        left=None,
        right=None,
        value=None
    ):
        self.feature_index = feature_index
        self.threshold = threshold
        self.is_numeric = is_numeric
        self.left = left
        self.right = right
        self.value = value


# =========================
# Decision Tree Classifier
# =========================
class DecisionTreeClassifierScratch:
    def __init__(self, max_depth=None, min_samples_split=2, min_gain=1e-6):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_gain = min_gain
        self.root = None
        self.feature_types = None

    # --------- PUBLIC API ---------
    def fit(self, X, y, feature_types):
        self.feature_types = feature_types
        self.root = self._build_tree(X, y, depth=0)

    def predict(self, X):
        return np.array([self._predict_one(x, self.root) for x in X])

    # --------- CORE TREE LOGIC ---------
    def _build_tree(self, X, y, depth):
        n_samples = X.shape[0]
        n_classes = len(np.unique(y))

        # Stopping conditions
        if (
            n_classes == 1
            or n_samples < self.min_samples_split
            or (self.max_depth is not None and depth >= self.max_depth)
        ):
            return DecisionTreeNode(value=self._majority_class(y))

        feature_idx, threshold, gain = self._best_split(X, y)
        if feature_idx is None or gain < self.min_gain:
            return DecisionTreeNode(value=self._majority_class(y))

        is_numeric = self.feature_types[feature_idx] == "numerical"
        X_left, y_left, X_right, y_right = self._split_dataset(
            X, y, feature_idx, threshold, is_numeric
        )

        left_child = self._build_tree(X_left, y_left, depth + 1)
        right_child = self._build_tree(X_right, y_right, depth + 1)

        return DecisionTreeNode(
            feature_index=feature_idx,
            threshold=threshold,
            is_numeric=is_numeric,
            left=left_child,
            right=right_child
        )

    def _predict_one(self, x, node):
        if node.value is not None:
            return node.value

        val = x[node.feature_index]
        if node.is_numeric:
            next_node = node.left if val <= node.threshold else node.right
        else:
            next_node = node.left if val == node.threshold else node.right

        return self._predict_one(x, next_node)

    # --------- SPLITTING LOGIC ---------
    def _best_split(self, X, y):
        n_samples, n_features = X.shape
        best_gain = 0.0
        best_feature = None
        best_threshold = None

        for feature_idx in range(n_features):
            feature_values = X[:, feature_idx]
            f_type = self.feature_types[feature_idx]

            if f_type == "numerical":
                unique_vals = np.unique(feature_values)
                if len(unique_vals) <= 1:
                    continue

                thresholds = (unique_vals[:-1] + unique_vals[1:]) / 2
                for thr in thresholds:
                    _, y_left, _, y_right = self._split_dataset(
                        X, y, feature_idx, thr, is_numeric=True
                    )
                    gain = self._information_gain(y, y_left, y_right)
                    if gain > best_gain:
                        best_gain = gain
                        best_feature = feature_idx
                        best_threshold = thr

            else:  # categorical
                for cat in np.unique(feature_values):
                    _, y_left, _, y_right = self._split_dataset(
                        X, y, feature_idx, cat, is_numeric=False
                    )
                    gain = self._information_gain(y, y_left, y_right)
                    if gain > best_gain:
                        best_gain = gain
                        best_feature = feature_idx
                        best_threshold = cat

        return best_feature, best_threshold, best_gain

    def _split_dataset(self, X, y, feature_index, threshold, is_numeric):
        if is_numeric:
            left_mask = X[:, feature_index] <= threshold
        else:
            left_mask = X[:, feature_index] == threshold

        right_mask = ~left_mask
        return X[left_mask], y[left_mask], X[right_mask], y[right_mask]

    # --------- IMPURITY MEASURES ---------
    def _entropy(self, y):
        values, counts = np.unique(y, return_counts=True)
        probs = counts / counts.sum()
        return -np.sum(probs * np.log2(probs + 1e-9))

    def _information_gain(self, parent_y, left_y, right_y):
        if len(left_y) == 0 or len(right_y) == 0:
            return 0.0

        n = len(parent_y)
        parent_entropy = self._entropy(parent_y)
        child_entropy = (
            (len(left_y) / n) * self._entropy(left_y)
            + (len(right_y) / n) * self._entropy(right_y)
        )
        return parent_entropy - child_entropy

    def _majority_class(self, y):
        values, counts = np.unique(y, return_counts=True)
        return values[np.argmax(counts)]
