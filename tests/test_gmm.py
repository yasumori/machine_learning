#!/usr/bin/env python3

import sys
import unittest
sys.path.append('code')

import numpy as np

from gmm import MGaussian, GMM

class TestMultivariateGaussian(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls._mgaussian = MGaussian(
                mean=np.array([0, 0]),
                covariance=np.array([1, 1]))

        cls._gmm = GMM(2)
        mean = np.array([0, 0])
        cov = np.array([1, 1])
        g1 = MGaussian(mean=mean, covariance=cov)
        g2 = MGaussian(mean=mean, covariance=cov)
        cls._gmm.gaussians = [g1, g2]
        cls._gmm.mixture_weights = np.array([0.5, 0.5])

    def test_compute_prob(self):
        data = np.array(
                [[0, 0],
                 [1, 1],
                 [2, 2]])
        out = self._mgaussian.compute_prob(data)
        exp = [0.15915494, 0.05854983, 0.00291502]
        np.testing.assert_array_almost_equal(out, exp)

    def test_gmm_compute_prob(self):
        data = np.array(
                [[0, 0],
                 [1, 1],
                 [2, 2]])
        out = self._gmm.compute_probs(data)
        exp = [[0.07957747, 0.02927492, 0.0014575],
               [0.07957747, 0.02927492, 0.0014575]]
        np.testing.assert_array_almost_equal(out, exp)

    def test_gmm_train(self):
        # prepare separable data
        data = np.array(
                [[1, 1],
                 [2, 2],
                 [1, 2],
                 [2, 1],
                 [-1, -1],
                 [-2, -2],
                 [-1, -2],
                 [-2, -1]])
        l = self._gmm.run_EM(data)
        # the model mostly converges to this likelihood
        # but sometimes not due to random parameter initialisation
        exp = -17.157839086719697
        self.assertEqual(l[-1], exp)

if __name__=="__main__":
    unittest.main()
