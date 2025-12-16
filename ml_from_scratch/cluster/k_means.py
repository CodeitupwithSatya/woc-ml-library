import numpy as np
import matplotlib.pyplot as plt
from ml_from_scratch.utils.distance import euclidean_distance


class KMeansFromScratch:
    def __init__(self, k=5, max_iters=100, tol=1e-4, plot_steps=False):
        self.k = k
        self.max_iters = max_iters
        self.tol = tol
        self.plot_steps = plot_steps

    def fit(self, X):
        self.X = X
        self.n_samples, self.n_features = X.shape

        # initialize centroids randomly
        random_idxs = np.random.choice(self.n_samples, self.k, replace=False)
        self.centroids = X[random_idxs]

        for _ in range(self.max_iters):
            # assign clusters
            self.clusters = self._create_clusters(self.centroids)

            # compute new centroids
            centroids_old = self.centroids.copy()
            self.centroids = self._get_centroids(self.clusters)

            # check convergence
            if self._is_converged(centroids_old, self.centroids):
                break

            if self.plot_steps:
                self.plot()

        return self

    def predict(self, X):
        labels = []
        for x in X:
            idx = self._closest_centroid(x, self.centroids)
            labels.append(idx)
        return np.array(labels)

    def fit_predict(self, X):
        self.fit(X)
        return self._get_cluster_labels(self.clusters)

    # -------- internal helpers -------- #

    def _create_clusters(self, centroids):
        clusters = [[] for _ in range(self.k)]
        for idx, sample in enumerate(self.X):
            centroid_idx = self._closest_centroid(sample, centroids)
            clusters[centroid_idx].append(idx)
        return clusters

    def _closest_centroid(self, sample, centroids):
        distances = [euclidean_distance(sample, c) for c in centroids]
        return np.argmin(distances)

    def _get_centroids(self, clusters):
        centroids = np.zeros((self.k, self.n_features))
        for idx, cluster in enumerate(clusters):
            if cluster:
                centroids[idx] = np.mean(self.X[cluster], axis=0)
        return centroids

    def _is_converged(self, old, new):
        distances = [euclidean_distance(old[i], new[i]) for i in range(self.k)]
        return sum(distances) < self.tol

    def _get_cluster_labels(self, clusters):
        labels = np.empty(self.n_samples)
        for cluster_idx, cluster in enumerate(clusters):
            for sample_idx in cluster:
                labels[sample_idx] = cluster_idx
        return labels.astype(int)

    def plot(self):
        # PCA for visualization (same logic you wrote)
        cov = np.cov(self.X.T)
        eigvals, eigvecs = np.linalg.eigh(cov)
        idx = eigvals.argsort()[::-1]
        W = eigvecs[:, idx[:2]]

        X_pca = self.X @ W
        centroids_pca = self.centroids @ W

        plt.figure(figsize=(8, 6))
        for i, cluster in enumerate(self.clusters):
            pts = X_pca[cluster]
            plt.scatter(pts[:, 0], pts[:, 1], label=f'Cluster {i}')
        plt.scatter(centroids_pca[:, 0], centroids_pca[:, 1],
                    c='black', marker='x', s=200)
        plt.legend()
        plt.show()
