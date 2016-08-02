#!/usr/bin/env python3

from copy import deepcopy
import unittest

import numpy as np

from k_means import KMeans

class TestKMeans(unittest.TestCase):
    def setUp(self):
        self._kmeans = KMeans(2) # n_clusters

    def test_get_intial_centroids(self):
        data = np.array([[1, 1], [0, 0], [-1, -1], [2, 2]])
        data = self._kmeans._get_initial_centroids(data)
        # random seed chooses the same centroids
        exp_data = np.array([[0, 0], [2, 2]])
        self.assertTrue(np.array_equal(data, exp_data))

        cent1, cent2 = [cluster.centroid for cluster in self._kmeans.clusters]
        exp_cent1 = np.array([1, 1])
        exp_cent2 = np.array([-1, -1])

        self.assertTrue(np.array_equal(cent1, exp_cent1))
        self.assertTrue(np.array_equal(cent2, exp_cent2))

    def test_choose_cluster(self):
        self._kmeans.clusters.append(self._kmeans.Cluster
                (np.array([1, 1]), initial=True))
        self._kmeans.clusters.append(self._kmeans.Cluster
                (np.array([-1, -1]), initial=True))
        self._kmeans._choose_cluster(np.array([1, 0]))
        self._kmeans._choose_cluster(np.array([-1, -2]))
 
        data_points1 = self._kmeans.clusters[0].data_points
        exp1 = np.array([[1, 1], [1, 0]])
        data_points2 = self._kmeans.clusters[1].data_points
        exp2 = np.array([[-1, -1], [-1, -2]])

        self.assertTrue(np.array_equal(data_points1, exp1))
        self.assertTrue(np.array_equal(data_points2, exp2))

    def test_squared_euclidian_dist(self):
        x1, y1 = (0, 0) #0
        x2, y2 = (np.array([1, 2, 3]), np.array([3, 2, 1])) #8
        x3, y3 = (np.array([[1, 2], [1, 2]])), np.array([[1, 1], [1, 1]])

        exp1 = 0
        res1 = self._kmeans._squared_euclidian_dist(x1, y1)
        exp2 = 8
        res2 = self._kmeans._squared_euclidian_dist(x2, y2)
        exp3 = 2
        res3 = self._kmeans._squared_euclidian_dist(x3, y3)

        self.assertEqual(res1, exp1)
        self.assertEqual(res2, exp2)
        self.assertEqual(res3, exp3)

    def test_is_finish_success(self):
        self._kmeans.clusters.append(self._kmeans.Cluster(np.array([1, 1])))
        self._kmeans.clusters.append(self._kmeans.Cluster(np.array([-1, -1])))
        self._kmeans.clusters[0].data_points = np.array([[1, 1], [1, 0]])
        self._kmeans.clusters[1].data_points = np.array([[1, 1], [-1, -2]])
        self._kmeans.prev_clusters = deepcopy(self._kmeans.clusters)

        res = self._kmeans._is_finish()
        exp = 1
        self.assertEqual(res, exp)

    def test_is_finish_fail(self):
        self._kmeans.clusters.append(self._kmeans.Cluster(np.array([1, 1])))
        self._kmeans.clusters.append(self._kmeans.Cluster(np.array([-1, -1])))
        self._kmeans.clusters[0].data_points = np.array([[1, 1], [1, 0]])
        self._kmeans.clusters[1].data_points = np.array([[1, 1], [-1, -2]])
        self._kmeans.prev_clusters = deepcopy(self._kmeans.clusters)
        self._kmeans.clusters[0].data_points = np.array([[2, 1], [1, 0]])

        res = self._kmeans._is_finish()
        exp = 0
        self.assertEqual(res, exp)

    def test_update_centroids_and_data(self):
        self._kmeans.clusters.append(self._kmeans.Cluster(np.array([1, 1])))
        self._kmeans.clusters.append(self._kmeans.Cluster(np.array([-1, -1])))
        self._kmeans.clusters[0].data_points = np.array([[1, 1], [1, 0]])
        self._kmeans.clusters[1].data_points = np.array([[1, 1], [-1, -2]])
        self._kmeans.prev_clusters = deepcopy(self._kmeans.clusters)

        data = self._kmeans._update_centroids_and_data()
        res_centroid1 = self._kmeans.clusters[0].centroid
        res_centroid2 = self._kmeans.clusters[1].centroid
        exp_centroid1 = np.array([1., 0.5])
        exp_centroid2 = np.array([0., -0.5])
        exp_data = np.array([[1, 1], [1, 0], [1, 1], [-1, -2]])

        self.assertTrue(np.array_equal(res_centroid1, exp_centroid1))
        self.assertTrue(np.array_equal(res_centroid2, exp_centroid2))
        self.assertTrue(np.array_equal(data, exp_data))

if __name__ == "__main__":
    unittest.main()
