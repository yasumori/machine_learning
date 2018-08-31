#!/usr/bin/env python3

import numpy as np

class MGaussian:
    """
    Multivariate Gaussian
    """
    #TODO: implement full covariance
    def __init__(self, mean=None, covariance=None):
        self.mean = mean
        self.covariance = covariance

        self.cov_type = None
        self.det = None
        self.inverse = None

        # if Gaussian is initialized with covariance
        if isinstance(covariance, np.ndarray):
            self.cov_type = self.check_cov_type()
            self.det, self.inverse = self.compute_det_and_inverse()

    def check_cov_type(self):
        if len(self.covariance.shape) == 1:
            return 'diagonal'
        elif len(self.covariance.shape) == 2:
            return 'full'
        else:
            raise ValueError('invalid shape of covariance matrix')

    def compute_det_and_inverse(self):
        # covariance is diagonal:
        if self.cov_type == 'diagonal':
            det = np.prod(self.covariance)
            inv = 1.0 / self.covariance
            return det, inv
        else:
            raise NotImplementedError('full covariance not yet')

    def compute_log_prob(self, data):
        if not isinstance(self.mean, np.ndarray) or \
                not isinstance(self.covariance, np.ndarray):
            raise ValueError('parameters not set yet')

        n_dim = self.mean.shape[0]

        t1 = (-n_dim / 2) * np.log(2 * np.pi)
        t2 = -0.5 * np.log(self.det)
        tmp = data - self.mean
        t3 = -0.5 * np.sum(tmp * self.inverse * tmp, axis=1)
        return t1 + t2 + t3

    def compute_prob(self, data):
        return np.exp(self.compute_log_prob(data))

    def compute_covariance(self, data, cov_type='diagonal'):
        if cov_type == 'diagonal':
            self.cov_type = cov_type
            self.covariance = np.sum((data-self.mean)**2, axis=0) / len(data)
            self.det, self.inverse = self.compute_det_and_inverse()
        else:
            raise NotImplementedError('full covariance not yet')

    def compute_mean(self, data):
        self.mean = np.mean(data, axis=0)

    def set_covariance(self, cov):
        self.covariance = cov
        self.cov_type = self.check_cov_type()
        self.det, self.inverse = self.compute_det_and_inverse()

class GMM:
    """
    Gaussian Mixture Model
    """
    #TODO: k-means initialization
    def __init__(self, n_components):
        self.K = n_components

        self.gaussians = [MGaussian() for i in range(n_components)]
        self.mixture_weights = np.zeros(n_components)

    def compute_log_likelihood(self, data):
        res = np.zeros((self.K, len(data)))
        for i, gaussian in enumerate(self.gaussians):
            probs = gaussian.compute_log_prob(data)
            res[i] = probs + np.log(self.mixture_weights[i])
        return res

    def run_EM(self, data, stop=0.001, init='random'):
        if self.K > len(data):
            raise ValueError(
                    'data points fewer than number of mixture components')

        if init == 'random':
            self.random_init(data)

        self.gaussians[0].mean = data[0]
        self.gaussians[0].compute_covariance(data)
        self.gaussians[1].mean = data[2]
        self.gaussians[1].compute_covariance(data)

        print([g.mean for g in self.gaussians])
        print([g.covariance for g in self.gaussians])
        L_history = [np.sum(self.compute_log_likelihood(data), axis=1)]

        i=0
        while True:
            i+=1
            print(i)
            #print([g.mean for g in self.gaussians])
            #print(self.mixture_weights)
            # E-step
            L = np.exp(self.compute_log_likelihood(data))
            gamma = L / np.sum(L, axis=0)
            summed_gamma = np.sum(gamma, axis=1)

            # M-step
            for k in range(self.K):
                kth_gaussian = self.gaussians[k]
                kth_sum = summed_gamma[k]
                # new mean
                mu = np.sum((gamma[k][:, np.newaxis] * data), axis=0) / kth_sum
                print(mu)
                kth_gaussian.mean = mu
                # new covariance
                tmp = (data - mu) ** 2
                cov = np.sum((gamma[k][:, np.newaxis] * tmp), axis=0) / kth_sum
                print(cov)
                kth_gaussian.set_covariance(cov)
                print(kth_gaussian.det)
                print(kth_gaussian.inverse)
            # new mixture weight
            print(self.get_params())
            self.mixture_weights = np.sum(gamma, axis=1) / np.sum(gamma)

            new_L = np.sum(self.compute_log_likelihood(data), axis=1)
            #TODO: THIS IS WRONG in LOG IT WOULDN'T WORK
            if i == 5:
            #if new_L - L_history[-1] <= np.log(stop):
                L_history.append(new_L)
                break
            else:
                L_history.append(new_L)
        return L_history

    def get_params(self):
        mu = [g.mean for g in self.gaussians]
        cov = [g.covariance for g in self.gaussians]
        return mu, cov

    def random_init(self, data):
        idxs = list(range(len(data)))
        centers = np.random.choice(idxs, size=self.K, replace=False)

        for center, gaussian in zip(centers, self.gaussians):
            gaussian.mean = data[center]
            gaussian.compute_covariance(data)

        self.mixture_weights = np.ones(self.K) / self.K
