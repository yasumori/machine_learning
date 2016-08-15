#!/usr/bin/env python3

import numpy as np

class LinearRegression:
    def __init__(self, n_in):
        """initialise a bias and weights with 0
           n_in: the number of feature variables for each data point
        """
        self.W = np.zeros(n_in)
        self.b = np.zeros(1)
        self.errors = list()
        self._is_univariate = True if n_in == 1 else False

    def train(self, data, n_iter=1000, lr=0.01):
        """train a linear regression model with gradient descent
           data: a tuple of data points and labels
           n_iter: the number of iterations
           lr: learning rate
        """
        x, y = data
        # initial error
        pred = self.predict(x)
        self.errors.append(self._sum_squared_error(pred, y))
        for i in range(n_iter):
            # compute gradient for W and b
            if self._is_univariate:
                #grad_W = lr * ((pred - y) * x).mean()
                grad_W = lr * (1 / len(y)) * sum((pred  - y) * x)
            else:
                grad_W = lr * (1 / len(y)) * (np.dot(pred - y, x))
            grad_b = lr * (1 / len(y)) * sum(pred - y)
            # update parameters
            self.W = self.W - grad_W
            self.b = self.b - grad_b

            pred = self.predict(x)
            self.errors.append(self._sum_squared_error(pred, y))

    def _is_converged(self):
        pass

    def _mean_sum_squared_error(self, pred, y):
        """compute the error of predicted values"""
        #np.mean((pred - y) ** 2)
        return (1 / len(y)) * sum((pred - y) ** 2)

    def predict(self, x):
        if self._is_univariate:
            return self.W * x + self.b
        return np.dot(self.W, x.T) + self.b
