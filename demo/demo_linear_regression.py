#!/usr/bin/env python3

import sys
sys.path.append('code')

import matplotlib.pyplot as plt
import numpy as np

from linear_regression import LinearRegression

# x: distance (km) from Dublin to city X
# y: Ryanair flight price (Euro) in next 3 months checked on 10/08/2018
data = [(258.22, 12.43),  # Glasgow
        (315.95, 14.19),  # Birmingham
        (335.58, 14.19),  # Bristol
        (474.87, 16.51),  # London Stansted
        (805.74, 18.19),  # Brussels
        (972.3, 19.19),   # Cologne
        (762.47, 20.19),  # Amsterdam
        (1085.37, 20.19), # Hamburg
        (1734.6, 22.19),  # Bratislava
        (1374.23, 22.19), # Munich
        (1630.8, 24.19),  # Bologna
        (1477.31, 24.19), # Barcelona Girona
        (299.56, 14.19),  # Edinburgh
        (218.72, 14.19),  # Liverpool
        (268.82, 14.19),  # Manchester
        (832.17, 18.19),  # Eindhoven
        (1006.08, 20.19), # Bremen
        (1616.25, 20.19), # Bydgoszcz
        (1972.53, 20.19), # Lublin
        (972.66, 20.19),  # Luxembourg
        (1222.2, 20.19),  # Stuttgart
        (1190.54, 22.19), # Basel
        (1764.8, 22.19),  # Lodz
        (1566.51, 22.19), # Poznan
        (1838.29, 22.19)  # Warsaw Modlin
        ]

testing = (1642.58, 22.19) # Gdansk

def display_data(x, y, model):
    dummy_x = list(range(200, 2100, 100))
    dummy_y = model.W * dummy_x + model.b

    plt.scatter(x, y)
    plt.plot(dummy_x, dummy_y)

    # automatically close plot after 2 seconds
    plt.show(block=False)
    plt.pause(2)
    plt.close()

if __name__=="__main__":
    x = np.array([d[0] for d in data])
    y = np.array([d[1] for d in data])

    model = LinearRegression(1)
    # without this initial parameter, model couldn't be optimized...
    model.b = 17
    display_data(x, y, model)

    # learning rate higher than this would increase error after each iter...
    # perhaps, data is not suitable for linear slope and intercept
    # or, there are too few data points
    model.train(x, y, n_iter=10, lr=0.000001)
    display_data(x, y, model)

    test_x, test_y = testing
    pred = model.predict(test_x)
    print('Dist from Dublin to Gdansk: {}\n'
            'answer  {}\n'
            'predict {}'.format(test_x, test_y, pred))
