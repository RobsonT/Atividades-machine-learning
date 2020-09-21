import numpy as np

class KNNClassifier():
    def __init__(self, _k_neighbours):
        """Initialize KNN class

        Args:
            _k_neighbours (Int): number of neghbours to be considered.
        """
        self._estimator_type = "classifier"
        self._k_neighbours = _k_neighbours

    def euclidian_distance(self, row1, row2):
        return np.sqrt(((row1 - row2)**2).sum())

    def fit(self, X, y):
        self.sample_features = X
        self.sample_targets = y

    def get_distances(self, X):
        distances = np.array([])
        for row in self.sample_features:
            distances = np.append(distances, self.euclidian_distance(X, row))

        return distances

    def get_neighbours(self, X):
        distances = self.get_distances(X)
        distances_indices = distances.argsort(axis=0)
        return self.sample_targets[distances_indices][:self._k_neighbours]

    def predict(self, X):
        predicted = np.array([])
        for x in X:
            neighbours = self.get_neighbours(x)
            targets = np.unique(neighbours)
            targets_count = np.array([])
            for label in targets:
                targets_count = np.append(targets_count, (neighbours == label).sum())
            target_index = np.argmax(targets_count)
            predicted = np.append(predicted, neighbours[target_index])
        return predicted

class MLP():
    
    def __init__(self, _layer_size, _epochs, _learning_rate):
        self._estimator_type = "classifier"
        self._layer_size = _layer_size
        self._epochs = _epochs
        self._learning_rate = _learning_rate

    def apply_activation(self, X):
        return 1.0 / (1 + np.exp(-1 * X))

    def apply_derivative_activation(self, x):
        activation = self.apply_activation(x)
        return activation * (1 - activation)

    def update_weights(self, X, y):
        y_pred = self.predict(X[:, 1:])
        error = y - y_pred

        result = np.array([])
        for value in self.u_output:
            result = np.append(result, self.apply_derivative_activation(value))

        deltak = error * result 

        result = np.array([])
        for row in self.u_hidden:
            for value in row:
                result = np.append(result, self.apply_derivative_activation(value))
        result = result.reshape(self.u_hidden.shape)

        deltai = np.array([])
        for i in range(X.shape[0]):
            deltai = np.append(deltai, result[i] * (deltak[i] * self.m).sum())
        deltai = deltai.reshape(X.shape[0], self._layer_size)
                 
        self.m = self.m + (self._learning_rate * (deltak.T @ self.z))
        self.w = self.w + (self._learning_rate * (deltai.T @ X))
    
    def fit(self, X, y):
        bias = -1 * np.ones((X.shape[0],1))
        X = np.hstack((bias,X))

        self.w = np.random.rand(self._layer_size, X.shape[1])
        self.m = np.random.rand(1, self._layer_size+1)

        for _ in range(self._epochs):
            self.update_weights(X, y)

    def predict(self, X):
        bias = -1 * np.ones((X.shape[0],1))
        X = np.hstack((bias,X))

        self.u_hidden = X @ self.w.T

        self.z = np.array([])
        for row in self.u_hidden:
            for value in row:
                result = self.apply_activation(value)
                self.z = np.append(self.z, result)
        self.z = self.z.reshape(self.u_hidden.shape)
        self.z = np.hstack((bias,self.z))

        self.u_output = self.z @ self.m.T

        self.y_pred = np.array([])
        for value in self.u_output:
            result = self.apply_activation(value)
            if result > 0.5:
                self.y_pred = np.append(self.y_pred, 1)
            else:
                self.y_pred = np.append(self.y_pred, 0)
        return self.y_pred


 
