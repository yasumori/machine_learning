#!/usr/bin/env python3

import sys
sys.path.append('code')

import numpy as np

from gmm import MGaussian, GMM

if __name__=="__main__":
    g = MGaussian(
            mean=np.array([0,0]),
            covariance=np.array([1, 1])
            )

    data = np.array([
        [0, 0],
        [1, 1],
        [2, 2]])

    # should get: 0.15915... 0.05854... 0.00291...
    g.compute_prob(data)

    gmm = GMM(2)
    gmm.random_init(data)
    for gaussian in gmm.gaussians:
        gaussian.mean = np.array([0, 0])
        gaussian.covariance = np.array([1, 1])
        gaussian.det, gaussian.inverse = gaussian.compute_det_and_inverse()


    g1 = MGaussian(
            mean=np.array([0,1]),
            covariance=np.array([1,2])
            )

    g2 = MGaussian(
            mean=np.array([2, 3]),
            covariance=np.array([2, 2]))

    gmm2 = GMM(2)
    gmm2.gaussians = [g1, g2]
    gmm2.mixture_weights = np.array([0.5,0.5])
    data2 = np.array([[0,1], [2,3], [0,0], [1,1]])
