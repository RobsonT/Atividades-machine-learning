from sklearn.metrics import plot_confusion_matrix
import matplotlib.pyplot as plt
import numpy as np

class Plots():
    def __init__(self):
        pass

    def plot_confusion_matrix_(self, X, y, clf):
        """Plot the confusion matrix for X and y

        Args:
            X (numpy array): sample data
            y (numpy array): label data
            clf (Object): a classifier model
        """
        confusion_matrix = plot_confusion_matrix(clf, X, y)
        confusion_matrix.ax_.set_title("Matriz de confusão")
        plt.show()

    def plot_boundaries(self, X, y, clf):
        """plot the boundaries to X and y

        Args:
            X (numpy array): sample data
            y (numpy array): label data
            clf (Object): a classifier model
        """
        x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
        y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
        h = 1.5

        xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
        Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
        
        Z = Z.reshape(xx.shape)
        plt.figure(1, figsize=(4, 3))
        plt.pcolormesh(xx, yy, Z, cmap=plt.cm.Paired)
        
        plt.scatter(X[:, 0], X[:, 1], c=y, edgecolors='k', cmap=plt.cm.Paired)
        plt.xlabel('X1')
        plt.ylabel('X2')

        plt.xlim(xx.min(), xx.max())
        plt.ylim(yy.min(), yy.max())
        plt.xticks(())
        plt.yticks(())

        plt.show()
    
