#!/usr/bin/env python3

import numpy as np

from k_means import KMeans

# Pokemon heigh/weight
data = np.array([[0.4, 6.0],  # Pikachu
                 [0.7, 6.9],  # Bulbasaur
                 [0.6, 8.5],  # Charmander
                 [0.5, 9.0],  # Squirtle
                 [1.2, 36.0], # Slowpoke
                 [1.6, 78.5], # Slowbro
                 [1.1, 90.0], # Seel
                 [1.7, 120.0],# Dewgong
                 [2.2, 210.0],# Dragonite
                 [1.7, 55.4], # Articuno
                 [1.6, 52.6], # Zapdos
                 [2.0, 60.0]] # Moltres
                 )
if __name__ == "__main__":
    k_means = KMeans(2)
    k_means.train(data)
    k_means.report()
