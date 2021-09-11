#!/usr/bin/env python3

import numpy as np

class LogisticRegression:
    def __init__(self, n_in, n_out):
        """initialise biases and wrights with 0
           n_in: the number of feature variables for each data point
           n_out: the number of types of output labels
        """
        self.W = np.zeros((n_in, n_out))
        self.b = np.zeros(n_out)
        self.errors = list()

    def train(self, data, n_iter=1000, lr=0.01):
        """train a logistic regression classifier with gradient descent
           data: a tuple of data points and labels
                 x: a set of data points consisting of feature values
                 y: a set of labels ranging from 0 to n_out - 1
           n_iter: the number of iterations
           lr: learning rate
        """
        x, y = data
        # initial error
        self.errors.append(self._negative_log_likelihood(y))
        for i in range(n_iter):
            probs = self._softmax(x)
            grad_W, grad_b = self._gradient(x, y)
            self.W = self.W - (lr * grad_W)
            self.b = self.b - (lr * grad_b)
            self.errors.append(self._negative_log_likelihood(y))

    def _gradient(self, x, y):
        """compute gradient for b and W"""
        # - 1/m * Sigma^m_i=1 [x^i(1{y^i=k} - p(y^i=k|x^i; W))]
        n_prob = -self._softmax(x)
        n_prob[np.arange(len(self.W)), y] += 1
        grad_W = -np.dot(x.T, n_prob) / len(y)
        # taking a mean of n_prob and its neagtive is a gradient for bias
        grad_b = -n_prob.mean(axis=0)
        return (grad_W, grad_b)

    def _negative_log_likelihood(self, y):
        return -np.log(self.W[np.arange(len(self.W)), y]).mean()

    def _softmax(self, x):
        """compute probabilities of each class given data points"""
        return np.exp(np.dot(x, self.W) + self.b) / \
            np.exp(np.dot(x, self.W) + self.b).sum(axis=1, keepdims=True)

    def predict(self, x):
        return self._softmax(x).argmax(axis=1)
