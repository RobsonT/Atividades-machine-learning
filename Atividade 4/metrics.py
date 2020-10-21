import numpy as np

class Metrics():
    def __init__(self):
        pass

    def accuracy_score(self, y_true, y_pred):
        """Calculate the accuracy score for the data passed

        Args:
            y_true (numpy array): the real data
            y_pred (numpy array): the predicted data

        Returns:
            float: the accuracy score for the data passed
        """
        sample_length = y_true.shape[0]
        return (np.sum(y_true == y_pred) / sample_length)

    def shuffle(self, X, y):
        indices = np.arange(X.shape[0])
        np.random.shuffle(indices)
        return X[indices], y[indices]

    def k_fold(self, X, y, k, classifier):
        X, y = self.shuffle(X, y)

        subset_size = round(X.shape[0] / k)
        X_subsets = [X[i:i+subset_size, :] for i in range(0, X.shape[0], subset_size)]
        y_subsets = [y[i:i+subset_size] for i in range(0, y.shape[0], subset_size)]

        train_accuracy_list = np.array([])
        test_accuracy_list = np.array([])
        print("___________________________K-fold___________________________")
        for i in range(k):
            X_test = X_subsets[i]
            y_test = y_subsets[i]

            X_train = np.array([])
            y_train = np.array([])
            for j in range(k):
                if i != j:
                    X_train = np.append(X_train, X_subsets[j])
                    y_train = np.append(y_train, y_subsets[j])
                X_train = X_train.reshape(y_train.shape[0], X.shape[1])

            classifier.fit(X_train,y_train)
            y_test_pred = classifier.predict(X_test)
            y_train_pred = classifier.predict(X_train)

            test_accuracy = self.accuracy_score(y_test, y_test_pred)
            train_accuracy = self.accuracy_score(y_train, y_train_pred)

            train_accuracy_list = np.append(train_accuracy_list, train_accuracy)
            test_accuracy_list = np.append(test_accuracy_list, test_accuracy)
            print("___________________________Iteração {}___________________________".format(i+1))
            print('Acurácia para dados de treino: {}'.format(train_accuracy))
            print('Acurácia para dados de teste: {}'.format(test_accuracy))
        print('\n')
        print('Acurácia geral de treino: {}'.format(train_accuracy_list.mean()))
        print('Acurácia geral de teste: {}'.format(test_accuracy_list.mean()))
            