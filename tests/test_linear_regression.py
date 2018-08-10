#!/usr/bin/env python3

import sys
import unittest
sys.path.append('code')

import numpy as np

from linear_regression import LinearRegression

class TestLinearRegression(unittest.TestCase):
    def setUp(self):
        self.model_simple = LinearRegression(1)
        self.model_multiple = LinearRegression(2)

    def test_mean_squared_error(self):
        pred = np.array([0, 0])
        label = np.array([1, 1])
        err = self.model_simple._mean_squared_error(pred, label)
        exp = 1
        self.assertTrue(err, exp)

        label2 = np.array([2, 4])
        err2 = self.model_simple._mean_squared_error(pred, label2)
        exp2 = 10
        self.assertTrue(err2, exp2)

    def test_predict(self):
        # Both W and b are 0 after initialization
        x1 = np.array([1, 2])
        pred1 = self.model_simple.predict(x1)
        exp1 = np.array([0, 0])

        self.assertTrue(np.array_equal(pred1, exp1))

        x2 = np.array([[1,1],
                       [2,2]])
        pred2 = self.model_multiple.predict(x2)
        exp2 = np.array([0, 0])

        self.assertTrue(np.array_equal(pred2, exp2))

    def test_train(self):
        # y = x + 0
        x1 = np.array([2, 4])
        y1 = np.array([2, 4])
        m1 = LinearRegression(1)
        m1.train(x1, y1, n_iter=1, lr=0.1)

        # expected W and b after 1 iteration with lr 0.1
        exp_W1 = np.array([1.0])
        exp_b1 = 0.3
        self.assertTrue(np.array_equal(m1.W, exp_W1))
        self.assertAlmostEqual(m1.b[0], exp_b1)

        # y = x1 + x2 + 0
        x2 = np.array([[2, 2],
                       [4, 4]])
        y2 = np.array([4, 8])
        m2 = LinearRegression(2)
        m2.train(x2, y2, n_iter=1, lr=0.1)

        # expected W and b after 1 iteration with lr 0.1
        exp_W2 = np.array([2.0, 2.0])
        exp_b2 = 0.6
        self.assertTrue(np.array_equal(m2.W, exp_W2))
        self.assertAlmostEqual(m2.b[0], exp_b2)

if __name__=="__main__":
    unittest.main()
