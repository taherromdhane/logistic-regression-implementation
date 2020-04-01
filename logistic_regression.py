import numpy as np

class LogisticRegression :

    def __init__(self, learning_rate = 0.005, loss='binary_crossentropy') :

        self.learning_rate = learning_rate
        self.costs = []
        self.loss = loss
        self.history = {}

    # Utility functions

    def __sigmoid(self, z):

        # This function computes the sigmoid of z, a scalar or numpy array of any size.
        s = 1 / (1 + np.exp(-z))

        return s

    def __compute_accuracy(self, y_true, y_pred):

        # This function computes the accuracy of predictions
        return (100 - np.mean(np.abs(y_pred - y_true)) * 100)

    def __compute_loss(self, y_pred, y_true) :

        # This function computes the loss, given the predicted and true labels
        if self.loss == 'binary_crossentropy' :
            cost = -((1/self.batch_size) * np.sum(y_true * np.log(y_pred) + (1-y_true) * np.log(1-y_pred)))

        return cost

    # Main methods

    def __initialize_parameters(self, X, y, metrics) :

        self.dim = X.shape[0]
        self.batch_size = X.shape[1]
        self.w = np.zeros((self.dim, 1))
        self.dw = np.zeros([self.dim, 1])
        self.b = 0
        self.db = 0
        for metric in metrics :
            self.history['train_' + metric] = []
            self.history['val_' + metric] = []

    def __propagate(self, X, y) :

        y_pred = self.__sigmoid(np.dot(self.w.T, X) + self.b)
        cost = self.__compute_loss(y_pred, y)
        self.dw = (1/self.batch_size) * np.dot(X, (y_pred - y).T)
        self.db = (1/self.batch_size) * np.sum(y_pred - y)
        cost = np.squeeze(cost)

        return cost

    def __optimize(self, X_train, y_train, X_val, y_val, epochs, metrics, verbose) :

        for i in range(epochs) :
            cost = self.__propagate(X_train, y_train)
            self.w -= (self.learning_rate * self.dw)
            self.b -= (self.learning_rate * self.db)
            self.costs.append(cost)

            if verbose and i%verbose==0 :
                if X_val and y_val :
                    y_pred_train = predict(X_train)
                    y_pred_val = predict(X_val)

                    if 'accuracy' in metrics :

                        train_accuracy = self.__compute_accuracy(y_pred_train, y)
                        self.history['train_accuracy'].append(train_accuracy)

                        val_accuracy = self.__compute_accuracy(y_pred, y)
                        self.history['val_accuracy'].append(val_accuracy)

                    if 'loss' in metrics :

                        y_pred_train = predict(X_train)
                        train_loss = self.__compute_loss(y_pred_train, y)
                        self.history['train_loss'].append(train_loss)

                        y_pred_val = predict(X_val)
                        val_loss = self.__compute_loss(y_pred, y)
                        self.history['val_loss'].append(val_loss)

                to_print = "epoch " + str(i) + " ==============\n"
                to_print += "Cost : " + str(cost) + " , "
                to_print += ''.join([metric + " : " + str(self.history[metric][-1]) + " , " for metric in self.history.keys()])
                print(to_print)
                #print(self.w)
                #print(self.dw)


    def fit(self, X_train, y_train, X_val=None, y_val=None, epochs=100, metrics=[], verbose=None) :

        assert (not ((X_val and not y_val) or (not X_val and y_val)))
        assert (epochs > 0)

        self.__initialize_parameters(X_train, y_train, metrics)
        self.__optimize(X_train, y_train, X_val, y_val, epochs, metrics, verbose)

        return self.history


    def predict(self, X_test) :

        y_pred = np.zeros((1, self.batch_size))
        y_pred = self.__sigmoid(np.dot(self.w.T, X_test) + self.b)
        y_rounded = np.round(y_pred)
        return y_rounded