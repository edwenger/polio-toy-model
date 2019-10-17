import logging

import matplotlib.pyplot as plt
import numpy as np

from village import Village
from triangulate import get_neighbors_dict


def test_accumulate_susceptibles():
    
    v1 = Village(loc=(0.2, 0.3),
                 N=100,
                 vaccinated_fraction=0.2,
                 neighbors=set())
                
    v1.S = 0  # start from transiently low susceptibles
    n_generations = 26*300
                
    S_timeseries = []
    for t in range(n_generations):
        v1.update()
        S_timeseries.append(v1.S)

    plt.plot(range(n_generations), S_timeseries)


def test_transmit_infections():
    
    v0 = Village(loc=(0.2, 0.3),
                 N=10,
                 vaccinated_fraction=0.5,
                 neighbors=set())

    v0.challenge(2)

    generation = 0
    while v0.S > 0 and v0.infecteds:
        print('\n=========\ngeneration-%d\n=========' % generation)
        v0.update()
        v0.transmit()
        print(v0.summary())
        generation += 1


def test_transmit_neighbors():
    
    v0 = Village(loc=(0.2, 0.3),
                 N=10,
                 vaccinated_fraction=0.5)
        
    v1 = Village(loc=(0.2, 0.3),
                 N=10,
                 vaccinated_fraction=0.5)
        
    v0.neighbors = [v1]  # only v0-towards-v1 transmissions between villages
    
    v0.challenge(2)

    generation = 0
    while (v0.S + v1.S) > 0 and (v0.infecteds or v1.infecteds):
        print('\n=========\ngeneration-%d\n=========' % generation)
    
        for v in (v0, v1):
            v.update()
                
        for v in (v0, v1):
            v.transmit()
            print(v.summary())
        
        print('=========')

        generation += 1


def test_neighbor_topology():

    n_locations = 500

    locations = np.random.rand(n_locations, 2)
    tri, _neighbors = get_neighbors_dict(locations)

    _points = [tuple(p) for p in tri.points]
    neighbors = {}
    for k, v in _neighbors.items():
        neighbors[k] = [_points[i] for i in v]

    normalized_sizes = np.random.rand(n_locations)

    fig, ax = plt.subplots(figsize=(7, 7))

    ax.triplot(locations[:, 0], locations[:, 1], tri.simplices.copy(),
               color='darkgray', alpha=0.3, lw=1)
    ax.scatter(locations[:, 0], locations[:, 1],
               color='darkgray', s=100 * normalized_sizes, alpha=0.5)

    ax.plot(locations[0][0], locations[0][1], 'o', color='c')
    ax.scatter(*zip(*neighbors[0]), color='m')

    ax.set(aspect='equal', yticks=[], xticks=[])
    fig.set_tight_layout(True)


if __name__ == '__main__':

    logging.basicConfig(format='%(message)s', level=logging.DEBUG)

    # test_accumulate_susceptibles()

    # test_transmit_infections()
    
    # test_transmit_neighbors()

    test_neighbor_topology()

    plt.show()

