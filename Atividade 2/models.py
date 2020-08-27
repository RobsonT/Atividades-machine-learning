import numpy as np

class LogisticRegression():
    def __init__(self, learning_rate, epochs):
        self._estimator_type = "classifier"
        self.epochs_ = epochs
        self.learning_rate_ = learning_rate

    def calculate_gradient(self, X, y):
        y_pred = self.predict(X[:, 1:])
        error = y - y_pred
        n = X.shape[0]

        features_size = X.shape[1]
        
        gradient = np.array([])
        for i in range(features_size):
            gradient = np.append(gradient, (self.learning_rate_ * np.sum((error * X[:, i]), axis=0) / n))
        return gradient

    def update_weights(self, X, y):
        gradient = self.calculate_gradient(X, y)
        self.b = self.b + gradient

    def fit(self, X, y):
        ones = np.ones((X.shape[0], 1))
        X = np.c_[ones, X]

        self.b = np.zeros(X.shape[1])

        for _ in range(self.epochs_):
            self.update_weights(X, y)

    def predict(self, X):
        predicted = np.array([])
        for x in X:
            result = self.b[0] + (1 / (1 + np.exp(-1 * (self.b[1:].T @ x))))
            predicted = np.append(predicted,  0 if result < 0.5 else 1)
        return predicted

class GaussianNaiveBayes():
    
    def __init__(self):
        self._estimator_type = "classifier"
        self.theta_ = np.array([])
        self.covariance_ = np.array([])
        self.class_prior_ = np.array([])
        self.classes_ = np.array([])

    def calculate_mean(self, X):
        return X.sum(axis=0) / X.shape[0]

    def calculate_covariance(self, X, mean):
        n = X.shape[0]
        return ((X - mean).T @ (X - mean)) / (n - 1)

    def get_features_per_label(self, X, y, label):
        return np.array([x for x, class_ in zip(X, y) if class_ == label] )

    
    def fit(self, X, y):
        self.classes_, classes_count =  np.unique(y, return_counts=True)

        self.class_prior_ = (classes_count / classes_count.sum())

        self.theta_ = np.array([])
        self.covariance_ = np.array([])
        for label in self.classes_:
            features_per_label = self.get_features_per_label(X, y, label)

            features_mean =self.calculate_mean(features_per_label)
            self.theta_ = np.append(self.theta_, features_mean)
        
        self.theta_ = self.theta_.reshape(len(self.classes_), X.shape[1])

        self.covariance_ = self.calculate_covariance(X, np.mean(X, axis=0))
        
    def calculate_probability(self, X, mean, covariance):
        columns = X.shape[0]
        exponent = np.exp(-(1.0/2) * (X - mean).T @ np.linalg.inv(covariance) @ (X - mean))
        probability = exponent / (np.power(2*np.pi, columns / 2) * np.sqrt(np.linalg.det(covariance)))
        return probability

    def predict(self, X):
        predict = np.array([])
        for x_row in X:
            pred = np.array([])
            for i in range(0, len(self.classes_)):
                x_probability = self.calculate_probability(x_row, self.theta_[i], self.covariance_)
                pred = np.append(pred, x_probability * self.class_prior_[i]) 
            predict = np.append(predict, self.classes_[np.argmax(pred)])
        return predict

class QuadraticDiscrimimantAnalisys():
    
    def __init__(self):
        self._estimator_type = "classifier"
        self.theta_ = np.array([])
        self.covariance_ = np.array([])
        self.class_prior_ = np.array([])
        self.classes_ = np.array([])

    def calculate_mean(self, X):
        return X.sum(axis=0) / X.shape[0]

    def calculate_covariance(self, X, mean):
        n = X.shape[0]
        return ((X - mean).T @ (X - mean)) / (n - 1)

    def get_features_per_label(self, X, y, label):
        return np.array([x for x, class_ in zip(X, y) if class_ == label] )

    
    def fit(self, X, y):
        self.classes_, classes_count =  np.unique(y, return_counts=True)

        self.class_prior_ = (classes_count / classes_count.sum())

        self.theta_ = np.array([])
        self.covariance_ = np.array([])
        for label in self.classes_:
            features_per_label = self.get_features_per_label(X, y, label)

            features_mean =self.calculate_mean(features_per_label)
            self.theta_ = np.append(self.theta_, features_mean)

            features_covariance = self.calculate_covariance(features_per_label, features_mean)
            self.covariance_ = np.append(self.covariance_, features_covariance)
        
        self.theta_ = self.theta_.reshape(len(self.classes_), X.shape[1])
        self.covariance_ = self.covariance_.reshape(len(self.classes_), X.shape[1], X.shape[1])
        
    def calculate_probability(self, X, mean, covariance):
        columns = X.shape[0]
        exponent = np.exp(-(1.0/2) * (X - mean).T @ np.linalg.inv(covariance) @ (X - mean))
        probability = exponent / (np.power(2*np.pi, columns / 2) * np.sqrt(np.linalg.det(covariance)))
        return probability

    def predict(self, X):
        predict = np.array([])
        for x_row in X:
            pred = np.array([])
            for i in range(0, len(self.classes_)):
                x_probability = self.calculate_probability(x_row, self.theta_[i], self.covariance_[i])
                pred = np.append(pred, x_probability * self.class_prior_[i]) 
            predict = np.append(predict, self.classes_[np.argmax(pred)])
        return predict