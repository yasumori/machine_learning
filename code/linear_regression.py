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

    def train(self, x, y, n_iter=1000, lr=0.01, stopping=0.0001):
        """train a linear regression model with gradient descent
           x: ndarray storing data points of n_in features
           y: ndarray storing labels of x
           n_iter: the number of iterations
           lr: learning rate
           stopping: stop if margin of ith and i-1th errors is lower than this
        """
        # initial error
        pred = self.predict(x)
        self.errors.append(self._mean_squared_error(pred, y))
        for i in range(n_iter):
            # compute gradient for W and b

            # partial derivative of error with regard to W
            if self._is_univariate:
                grad_W = lr * (1 / len(y)) * sum((pred - y) * x)
            else:
                grad_W = lr * (1 / len(y)) * (np.dot(pred - y, x))
            # partial derivative of error with regard to b
            grad_b = lr * (1 / len(y)) * sum(pred - y)

            # update parameters
            self.W = self.W - grad_W
            self.b = self.b - grad_b

            pred = self.predict(x)
            error = self._mean_squared_error(pred, y)
            self.errors.append(error)

            if self._is_converged(error, stopping):
                print('Stop training at {} iter'.format(i+1))
                break

    def _is_converged(self, cur_error, stopping):
        # error progress too small
        if self.errors[-2] - cur_error < stopping:
            return True
        else:
            return False

    @staticmethod
    def _mean_squared_error(pred, y):
        """compute the error of predicted values"""
        return (1 / len(y)) * sum((pred - y) ** 2)

    def predict(self, x):
        if self._is_univariate:
            return self.W * x + self.b
        return np.dot(self.W, x.T) + self.b
