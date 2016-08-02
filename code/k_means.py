#!/usr/bin/env python3

from copy import deepcopy
from operator import attrgetter

import numpy as np

class KMeans:
    def __init__(self, n_clusters, random_seed=1006):
        self.n_clusters = n_clusters
        self.clusters = list()
        self.prev_clusters = None
        np.random.seed(random_seed)

    class Cluster:
        def __init__(self, centroid, initial=False):
            self.centroid = centroid
            self.data_points = list()
            if initial:
                self.data_points.append(centroid)

    def train(self, data):
        """produce n clusters given data"""
        data = self._get_initial_centroids(data)
        i = 0
        while True:
            i += 1
            print("\riteration {}".format(i))
            for data_point in data:
                self._choose_cluster(data_point)

            if self.prev_clusters:
                if self._is_finish():
                    break
            data = self._update_centroids_and_data()

    def _get_initial_centroids(self, data):
        """randomly choose n centroids"""
        for i in range(self.n_clusters):
            choice = np.random.randint(0, len(data))
            centroid = data[choice]
            data = np.delete(data, choice, axis=0)

            self.clusters.append(self.Cluster(centroid, initial=True))
        return data

    def _choose_cluster(self, data_point):
        """choose the nearest cluster for the given data point"""
        nearest = None
        for cluster in self.clusters:
            dist = self._squared_euclidian_dist(cluster.centroid, data_point)
            if nearest:
                if dist < nearest[1]:
                    nearest = (cluster, dist)
            else:
                nearest = (cluster, dist)
        chosen = nearest[0]
        chosen.data_points.append(data_point)

    @staticmethod
    def _squared_euclidian_dist(x, y):
        """function that computes the squared euclidian distance
           of two arguments
           possible type: int, float, numpy.ndarray or list of numpy.ndarray
        """
        if isinstance(x, (int, float)) and isinstance(y, (int, float)):
            return np.sum(abs(x - y)**2)
        elif isinstance(x, (np.ndarray)) and isinstance(y, (np.ndarray)):
            return np.sum(abs(x - y)**2)
        elif isinstance(x, list) and isinstance(y, list):
            return np.sum([abs(x - y)**2 for x, y in zip(x, y)])
        else:
            raise ValueError("data type of x and y not compatible")

    def _is_finish(self):
        """return 1 if the euclidian distance of previous data points and
           current data points is 0"""
        dist = 0
        for prev, curr in zip(self.prev_clusters, self.clusters):
            dist += self._squared_euclidian_dist(prev.data_points,
                    curr.data_points)
        return 1 if dist == 0 else 0

    def _update_centroids_and_data(self):
        """choose next centroids, and put all data back"""
        self.prev_clusters = deepcopy(self.clusters)
        self.clusters = list()
        data = np.array(self.prev_clusters[0].data_points)
        for i, cluster in enumerate(self.prev_clusters):
            data_points = cluster.data_points
            new_centroid = np.mean(data_points, axis=0)
            self.clusters.append(self.Cluster(new_centroid))
            if i > 0:
                data = np.vstack((data, data_points))
        return data

    def report(self):
        for i, cluster in enumerate(self.clusters):
            print("cluster{0}: {1}".format(i+1, cluster.data_points))
