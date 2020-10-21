from sklearn.preprocessing import StandardScaler
import numpy as np

class PCA():
    def __init__(self, n_components):
        self._n_components = n_components

    def calculate_covariance(self, X):
        n = X.shape[0]
        return ((X - np.mean(X, axis=0)).T @ (X - np.mean(X, axis=0))) / (n - 1)
    
    def fit(self, X):
        X = StandardScaler().fit_transform(X)
        covariance = self.calculate_covariance(X)
        values, vectors = np.linalg.eig(covariance)
        indices = (-values).argsort()[:self._n_components]
        self.w = vectors[:, indices]
        self.variance = np.array([])
        for i in indices:
            self.variance = np.append(self.variance, values[i] / values.sum())


    def transform(self, X):
        X = StandardScaler().fit_transform(X)
        return (self.w.T @ X.T).T