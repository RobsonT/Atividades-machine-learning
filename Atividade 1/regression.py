import numpy as np
import random

"""
file with all the regresssions classes
"""

class RLA():
    """
    Implementation of analitic linear regression
    """
    def __init__(self):
        pass
    
    def fit(self, x, y):
        """
        Traninig the linear regression

        Args:
            x (array): sample data
            y (array): label data
        """
        x_mean = np.mean(x)
        y_mean = np.mean(y)

        aux = (x -x_mean) * (y - y_mean)
        self.b1 = np.sum(aux) / np.sum((x - x_mean) ** 2)
        self.b0 = y_mean - self.b1 * x_mean
        
    def predict(self, x):
        """
        Predict tha label data

        Args:
            x (array): sample data

        Returns:
            array: label data for x
        """
        return self.b0 + (self.b1*x)

class RLGD():
    """
    Implementation of linear regression using gradient descent
    """
    def __init__(self, learning_rate=0.01, epochs = 100, save_epoch_mse=False):
        """
        Initialize RLGD class

        Args:
            learning_rate (float, optional): learning rate of gradient descent. 
            Defaults to 0.01.
            epochs (int, optional): quantity of epochs. Defaults to 100.
            save_epoch_mse (bool, optional): Save the mse of each epoch.
        """
        self.learning_rate = learning_rate
        self.b0 = random.random()
        self.b1 = random.random()
        self.epochs = epochs
        self.save_epoch_mse = save_epoch_mse
        self.mse = np.array([])

    def calculate_gradient_descent(self, x, y):
        """
        calculate the gradient descent

        Args:
            x (array): sample data
            y (array): label data
        """
        y_predicted = self.predict(x)
        error = y - y_predicted

        self.b0 = self.b0 + (self.learning_rate * np.mean(error))
        self.b1 = self.b1 + (self.learning_rate * np.mean(error * x))
        
        if self.save_epoch_mse:
            mse = np.mean((y - self.predict(x)) ** 2)
            self.mse = np.append(self.mse, mse)

    def fit(self, x, y):
        """
        Traninig the gradient descent linear regression

        Args:
            x (array): sample data
            y (array): label data
        """
        for _ in list(range(self.epochs)):
            self.calculate_gradient_descent(x, y)
        
    def predict(self, x):
        """
        Predict the label data

        Args:
            x (array): sample data

        Returns:
            array: label data for x
        """
        return self.b0 + (self.b1*x)

class MLR():
    """
    analitic multiple linear regression implementation
    """
    def __init__(self):
        pass
    
    def fit(self, x, y):
        """
        Train the model

        Args:
            x (array): sample data
            y (array): label data
        """
        ones = np.ones((x.shape[0],1))
        x = np.hstack((ones,x))

        aux = np.linalg.inv(np.matmul(x.T,x))
        self.b = np.matmul(np.matmul(aux,x.T),y)
        
    def predict(self, x):
        """
        Predict the label for x

        Args:
            x (array): sample data

        Returns:
            array: label data
        """
        y_predict = self.b[0]

        for i in range(1, self.b.shape[0]):
            y_predict += self.b[i]*x[:,i-1]

        return y_predict

class MLRGD():
    """
    multiple linear regression implementation using gradient descent
    """
    def __init__(self, learning_rate=0.01, epochs=1, save_epoch_mse=False):
        """
        Initialize MLRGD class

        Args:
            learning_rate (float, optional): learning rate of gradient descent. 
            Defaults to 0.01.
            epochs (int, optional): quantity of epochs. Defaults to 1.
        """
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.b = np.array([])
        self.save_epoch_mse = save_epoch_mse
        self.mse = np.array([])

    def calculate_gradient_descent(self,x, y):
        """
        calculate the gradient descent

        Args:
            x (array): sample data
            y (array): label data
        """
        y_predicted = self.predict(x)
        error = y - y_predicted

        features_size = x.shape[1]
        
        for i in range(features_size):
            self.b[i] = self.b[i] + (self.learning_rate * np.mean(error * x[:, i]))
            
        if self.save_epoch_mse:
            mse = np.mean((y - self.predict(x)) ** 2)
            self.mse = np.append(self.mse, mse)
    
    def fit(self, x, y):
        """
        Train the model

        Args:
            x (array): sample data
            y (array): label data
        """
        ones = np.ones((x.shape[0],1))
        x = np.hstack((ones,x))

        for _ in range(x.shape[1]):
            self.b = np.append(self.b, random.random())

        for _ in range(self.epochs):
            self.calculate_gradient_descent(x, y)
        
    def predict(self, x):
        """
        Predict the label for x

        Args:
            x (array): sample data

        Returns:
            array: label data
        """
        y_predict = self.b[0]

        for i in range(1, self.b.shape[0]):
            y_predict += self.b[i]*x[:,i-1]

        return y_predict

class MLRSGD():
    """
    multiple linear regression implementation using stochastic gradient descent
    """
    def __init__(self, learning_rate=0.01, epochs=100, save_epoch_mse=False):
        """
        Initialize MLREGD class

        Args:
            learning_rate (float, optional): learning rate of gradient descent. 
            Defaults to 0.01.
            epochs (int, optional): quantity of epochs. Defaults to 100.
        """
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.b = np.array([])
        self.save_epoch_mse = save_epoch_mse
        self.mse = np.array([])

    def calculate_stochastic_gradient_descent(self,x, y):
        """
        calculate the gradient descent

        Args:
            x (array): sample data
            y (array): label data
        """
        y_predicted = self.predict(x)
        error = y - y_predicted

        self.b = self.b + (self.learning_rate * (error @ x))
        
        if self.save_epoch_mse:
            mse = np.mean((y - self.predict(x)) ** 2)
            self.mse = np.append(self.mse, mse)
    
    def fit(self, x, y):
        """
        Train the model

        Args:
            x (array): sample data
            y (array): label data
        """
        ones = np.ones((x.shape[0],1))
        x = np.hstack((ones,x))

        for _ in range(x.shape[1]):
            self.b = np.append(self.b, random.random())

        for _ in range(self.epochs):
            self.calculate_stochastic_gradient_descent(x, y)
        
    def predict(self, x):
        """
        Predict the label for x

        Args:
            x (array): sample data

        Returns:
            array: label data
        """
        y_predict = self.b[0]

        for i in range(1, self.b.shape[0]):
            y_predict += self.b[i]*x[:,i-1]

        return y_predict

class Polynomial_regression():
    """
    Polynomial regression implementation
    """
    def __init__(self, degree=2):
        """
        Initialize polynomial regression class

        Args:
            degree (int, optional): degree of polynomial regression. Defaults to 2.
        """
        self.degree = degree

    
    def fit(self, x, y):
        """
        Train the model

        Args:
            x (array): sample data
            y (array): label data
        """
        x_transformed = x
        new_x = x
        
        for _ in range(2,self.degree + 1):
            new_x = new_x * x
            x_transformed = np.c_[x_transformed, new_x]

        self.model = MLR()
        self.model.fit(x_transformed, y)
        self.b = self.model.b
        
    def predict(self, x):
        """
        Predict the label for x

        Args:
            x (array): sample data

        Returns:
            array: label data
        """
        x_transformed = x
        new_x = x
        
        for _ in range(2,self.degree + 1):
            new_x = new_x * x
            x_transformed = np.c_[x_transformed, new_x]

        return self.model.predict(x_transformed)

class Regularized_multiple_linear_regression():
    """
    regularized multiple linear regression implementation using gradient descent
    """
    def __init__(self, learning_rate=0.01, epochs=100, _lambda = 1, save_epoch_mse=False):
        """
        Initialize MLRGD class

        Args:
            learning_rate (float, optional): learning rate of gradient descent. 
            Defaults to 0.01.
            epochs (int, optional): quantity of epochs. Defaults to 100.
        """
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.b = np.array([])
        self._lambda = _lambda
        self.save_epoch_mse = save_epoch_mse
        self.mse = np.array([])

    def calculate_gradient_descent(self,x, y):
        """
        calculate the gradient descent

        Args:
            x (array): sample data
            y (array): label data
        """
        y_predicted = self.predict(x)
        error = y - y_predicted
        
        features_size = x.shape[1]

        self.b0 = self.b0 + (self.learning_rate * np.mean(error))

        regularization_value = self._lambda * np.mean(self.b)
        
        for i in range(1, features_size):
            self.b[i] = self.b[i] + (self.learning_rate * (np.mean(error * x[:, i]) - regularization_value))
            
        if self.save_epoch_mse:
            mse = np.mean((y - self.predict(x)) ** 2)
            self.mse = np.append(self.mse, mse)
    
    def fit(self, x, y):
        """
        Train the model

        Args:
            x (array): sample data
            y (array): label data
        """
        self.b0 = random.random()
         
        for _ in range(x.shape[1]):
            self.b = np.append(self.b, random.random())

        for _ in range(self.epochs):
            self.calculate_gradient_descent(x, y)
        
    def predict(self, x):
        """
        Predict the label for x

        Args:
            x (array): sample data

        Returns:
            array: label data
        """
        y_predict = self.b0

        for i in range(self.b.shape[0]):
            y_predict += self.b[i]*x[:,i]

        return y_predict
