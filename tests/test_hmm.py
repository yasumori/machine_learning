#!/usr/bin/env python3

from copy import deepcopy
import sys
import unittest
sys.path.append('code')

import numpy as np

from hidden_markov_model import HMM

class TestHMM(unittest.TestCase):
    def setUp(self):
        # Jurafsky ice cream HMM
        self._ic_hmm = HMM(2, 3)

        # emission probs              1    2    3
        self._ic_hmm.Q = np.array([[0.2, 0.4, 0.4],  # Hot
                                   [0.5, 0.4, 0.1]]) # Cold

        # transition probs          q1   q2
        self._ic_hmm.A = np.array([[0.6, 0.3], # q1
                                  [0.4, 0.5]]) # q2

        # start state to {q1,q2}     q1   q2
        self._ic_hmm.q0 = np.array([0.8, 0.2])

        # from {q1,q2} to end state  q1   q2
        self._ic_hmm.qF = np.array([0.1, 0.1])

        # '3 1 3' as in the book
        self.ice_creams = [2, 0, 2]

        self.N = 2
        self.T = 3

    def test_forward(self):
        res = self._ic_hmm.forward(self.ice_creams)
        exp = np.array([[0.32, 0.04, 0.01808],
                        [0.02, 0.053, 0.00385]])
        self.compare_arrays(res, exp)

    def test_backward(self):
        res = self._ic_hmm.backward(self.ice_creams)
        exp = np.array([[0.00639, 0.027, 0.1],
                        [0.00741, 0.021, 0.1]])
        self.compare_arrays(res, exp)

    def test_likelihood(self):
        res = self._ic_hmm.compute_likelihood(self.ice_creams)
        exp = 0.002193
        self.assertAlmostEqual(res, exp)

    def test_viterbi(self):
        res = self._ic_hmm.viterbi(self.ice_creams)
        exp = np.array([[0.32, 0.0384, 0.009216],
                        [0.02, 0.048, 0.0024]])
        self.compare_arrays(res, exp)

    def test_decode(self):
        res = self._ic_hmm.decode(self.ice_creams)
        exp = np.array([0, 1, 0])
        self.assertTrue(np.array_equal(res, exp))

    def test_forward_backward(self):
        res_f, res_b = self._ic_hmm.forward_backward(self.ice_creams)
        exp_f = np.array([[0.32, 0.04, 0.01808],
                          [0.02, 0.053, 0.00385]])
        exp_b = np.array([[0.00639, 0.027, 0.1],
                          [0.00741, 0.021, 0.1]])
        self.compare_arrays(res_f, exp_f)
        self.compare_arrays(res_b, exp_b)

    def test_train(self):
        # needs another model to update parameters
        model = HMM(2, 3)
        model.Q = deepcopy(self._ic_hmm.Q)
        model.A = deepcopy(self._ic_hmm.A)
        model.q0 = deepcopy(self._ic_hmm.q0)
        model.qF = deepcopy(self._ic_hmm.qF)

        # model will overfit in ice_creams
        model.train([self.ice_creams])

        exp_Q = np.array([[0.0, 0.0, 1.0],
                          [1.0, 0.0, 0.0]])
        exp_A = np.array([[0.0, 0.5],
                          [1.0, 0.0]])
        exp_q0 = np.array([1.0, 0.0])
        exp_qF = np.array([0.5, 0.0])

        self.compare_arrays(model.Q, exp_Q)
        self.compare_arrays(model.A, exp_A)
        np.testing.assert_array_almost_equal(model.q0, exp_q0)
        np.testing.assert_array_almost_equal(model.qF, exp_qF)

    def compare_arrays(self, arr1, arr2):
        I, J = arr1.shape
        for i in range(I):
            for j in range(J):
                self.assertAlmostEqual(arr1[i, j], arr2[i, j])

if __name__=="__main__":
    unittest.main()
