import numpy as np
import math

class Kmeans():
    def __init__(self, _k):
        self._k = _k
    
    def euclidian_distance(self, row1, row2):
        return np.sqrt(((row1 - row2)**2).sum())

    def assign_group(self, X):
        self.classes = {}
        for i in range(self._k):
            self.classes[i] = np.zeros(X.shape[1])

        for x in X:
            distances = np.array([self.euclidian_distance(x, centroid) for centroid in self.centroids])
            label = np.argmin(distances)
            self.classes[label] = np.vstack((self.classes[label], x))

    def calculate_centroids(self, X):
        previous = self.centroids
        isOptimal = False
        while(not isOptimal):    
            countDifferents = 0
            for label in self.classes:
                self.centroids[label] = np.average(self.classes[label], axis = 0)

                if(not (previous[label] == self.centroids[label]).all()):
                    countDifferents += 1

            if(countDifferents == 0):
                isOptimal = True

            if(not isOptimal):
                self.assign_group(X)


    def fit(self, X):
        self.centroids = np.empty((self._k, X.shape[1]), float)
        for i in range(self._k):
            self.centroids[i] = np.array([np.random.randint(np.min(X[:, j]), np.max(X[:, j])) for j in range(X.shape[1])])

        self.assign_group(X)
        self.calculate_centroids(X)

    def predict(self, X):
        self.class_distance = np.array([])
        predicted = np.array([])
        for x in X:
            distances = np.array([self.euclidian_distance(x, centroid) for centroid in self.centroids])
            label = np.argmin(distances)
            self.class_distance = np.append(self.class_distance,np.min(distances))

            predicted = np.append(predicted, label)

        return predicted

class TreeNode:
    def __init__(self, data, feature):
        self.feature = feature
        self.value = data
        self.left = None
        self.right = None

class DecisionTreeClassifier():
    def __init__(self):
        self._estimator_type = "classifier"

    def count_unique(self, y):
        (unique, counts) = np.unique(y, return_counts=True)
        frequency = {}
        for i in range(len(unique)):
            frequency[unique[i]] = counts[i]
        return frequency

    def calculate_gini(self, X, y, feature, value):
        x = X[:, feature]
        left = {
            'total': 0
        }
        right = {
            'total': 0
        }
        
        for class_value in set(y):
            left[class_value] = 0
            right[class_value] = 0

        for i in range(len(x)):
            if x[i] > value:
                right['total'] += 1
                right[y[i]] += 1
            else:
                left['total'] += 1
                left[y[i]] += 1

        left_gini = 0
        right_gini = 0

        for class_value in set(y):
            if left['total'] > 0:
                left_gini += (left[class_value] / left['total'])  ** 2
            if right['total'] > 0:
                right_gini += (right[class_value] / right['total'])  ** 2
        
        left_gini = 1 - left_gini
        right_gini = 1 - right_gini

        class_frequency = self.count_unique(y)
        gini_root = 0
        for class_value in set(y):
            gini_root += (class_frequency[class_value] / y.shape[0]) ** 2

        return 1 - gini_root

    def build_decision_tree(self,X,y, features):
        if len(set(y)) == 1:
            value = y[0]
            return TreeNode(value, None)
            
        if X.shape[0] == 0 or X.shape[1] == len(features):
            class_frequency = self.count_unique(y)
            max_count = -math.inf
            value = None
            for i in set(y):
                if i in class_frequency:
                    if class_frequency[i] > max_count:
                        value = i
                        max_count = class_frequency[i]
            return TreeNode(value, None)

        max_gain = -math.inf
        final_feature = None
        final_value = None

        for feature in range(X.shape[1]):
            if (feature not in features):
                for value in np.unique(X[feature]):
                    current_gain = self.calculate_gini(X, y, feature, value)

                if current_gain > max_gain:
                    max_gain = current_gain
                    final_feature = feature
                    final_value = value

        left_y = y[X[:, final_feature] <= final_value]
        right_y = y[X[:, final_feature] > final_value]
        left_X = X[X[:, final_feature] <= final_value]
        right_X = X[X[:, final_feature] > final_value]
        features.append(final_feature)
        print(features)

        left = self.build_decision_tree(left_X, left_y, features)
        right = self.build_decision_tree(right_X, right_y, features)
        root = TreeNode(final_value, final_feature)
        root.left = left
        root.right = right
        return root
        

    def fit(self,X,y):
        features = []
        self.classes_ = np.unique(y)
        self.root = self.build_decision_tree(X, y, features)

    def predict(self, X):
        predict = np.array([])
        for x in X:
            root = self.root
            while(not (root.feature is None)):
                if(x[root.feature] > root.value):
                    root = root.right
                else:
                    root = root.left
            predict = np.append(predict, root.value)
        return predict