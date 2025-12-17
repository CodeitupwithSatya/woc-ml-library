import numpy as np
from collections import Counter
from copy import deepcopy
from .decision_tree import DecisionTreeClassifierScratch


class RandomForestClassifierScratch:

    def __init__(
        self,
        n_estimators=100,
        max_features="sqrt",
        max_depth=None,
        min_samples_split=2,
        min_gain=1e-6,
        bootstrap_size=None
    ):
        self.n_estimators = n_estimators
        self.max_features = max_features  # number of random features per tree
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_gain = min_gain
        self.bootstrap_size = bootstrap_size  # if None → n_samples

        self.trees = []
        self.feature_indices_used = []


    def _sample_bootstrap(self, X, y):
        n_samples = X.shape[0]

        size = self.bootstrap_size if self.bootstrap_size else n_samples

        idxs = np.random.choice(n_samples, size=size, replace=True)

        return X[idxs], y[idxs]


    def _get_random_features(self, n_features):
        """
        Returns list of randomly selected feature indices.
        """

        if isinstance(self.max_features, int):
            k = self.max_features

        elif self.max_features == "sqrt":
            k = int(np.sqrt(n_features))

        elif self.max_features == "log2":
            k = int(np.log2(n_features))

        else:
            k = n_features

        k = max(1, min(k, n_features))  # protection

        return np.random.choice(n_features, k, replace=False)


    def fit(self, X, y, feature_types):
        self.trees = []
        self.feature_indices_used = []

        n_samples, n_features = X.shape

        for _ in range(self.n_estimators):

            # bootstrap dataset
            X_sample, y_sample = self._sample_bootstrap(X, y)

            # random features
            feat_idxs = self._get_random_features(n_features)

            X_subset = X_sample[:, feat_idxs]
            feature_subset_types = [feature_types[i] for i in feat_idxs]

            # train decision tree
            tree = DecisionTreeClassifierScratch(
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                min_gain=self.min_gain
            )

            tree.fit(X_subset, y_sample, feature_subset_types)

            self.trees.append(tree)
            self.feature_indices_used.append(feat_idxs)


    def predict(self, X):
        tree_preds = []

        for tree, feat_idxs in zip(self.trees, self.feature_indices_used):
            p = tree.predict(X[:, feat_idxs])
            tree_preds.append(p)

        tree_preds = np.array(tree_preds)   # shape (n_trees, n_samples)

        # majority vote column-wise
        predictions = []

        for col in tree_preds.T:
            label = Counter(col).most_common(1)[0][0]
            predictions.append(label)

        return np.array(predictions)
    

class RandomForestRegressor:
    def __init__(
        self,
        n_estimators=100,
        max_features="sqrt",
        max_depth=None,
        min_samples_split=2,
        bootstrap=True,
        random_state=None,
    ):
        self.n_estimators = n_estimators
        self.max_features = max_features
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.bootstrap = bootstrap

        self.trees = []
        self.feature_subsets = []
        self.random_state = random_state
        if random_state:
            np.random.seed(random_state)

    def _get_n_features(self, n_features):
        if self.max_features == "sqrt":
            return int(np.sqrt(n_features))
        elif self.max_features == "log2":
            return int(np.log2(n_features))
        elif isinstance(self.max_features, float):  
            return int(n_features * self.max_features)
        elif isinstance(self.max_features, int):
            return self.max_features
        else:  
            return n_features  

    def _bootstrap_sample(self, X, y):
        n_samples = X.shape[0]
        idxs = np.random.choice(n_samples, n_samples, replace=True)
        return X[idxs], y[idxs]

    def fit(self, X, y, base_tree_class):
        """
        base_tree_class = your DecisionTreeRegressor class
        """
        n_samples, n_features = X.shape
        
        for _ in range(self.n_estimators):

            # 1. bootstrap sampling
            if self.bootstrap:
                sample_X, sample_y = self._bootstrap_sample(X, y)
            else:
                sample_X, sample_y = X, y

            # 2. feature subsampling
            n_feats = self._get_n_features(n_features)
            feat_idxs = np.random.choice(n_features, n_feats, replace=False)

            # 3. train tree on selected features
            tree = base_tree_class(
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                feature_indices=feat_idxs
            )

            tree.fit(sample_X[:, feat_idxs], sample_y)

            # store tree + feature subset
            self.trees.append(tree)
            self.feature_subsets.append(feat_idxs)

    def predict(self, X):

        tree_preds = np.zeros((self.n_estimators, X.shape[0]))

        for i, tree in enumerate(self.trees):
            feat_idxs = self.feature_subsets[i]
            preds = tree.predict(X[:, feat_idxs])
            tree_preds[i] = preds

        # mean of tree predictions
        return np.mean(tree_preds, axis=0)

