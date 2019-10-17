import logging

import numpy as np

from triangulate import get_neighbors_dict
from village import Village


class PowerLawPopulationDistribution(object):

    """
    Transform a numpy array of [0,1] normalized sizes
    to a power-law population distribution
    """

    def __init__(self, offset=1, scale=3):
        self.offset = offset
        self.scale = scale

    def __call__(self, x):
        return np.power(10, self.offset + self.scale * x)


def initialize_topology(n_locations=500):

    locations = np.random.rand(n_locations, 2)
    _, neighbors = get_neighbors_dict(locations)
    N = PowerLawPopulationDistribution()(np.random.rand(n_locations))
    vax = np.random.uniform(0.25, 0.75, size=n_locations)

    villages = [Village(loc=locations[i], N=N[i], vaccinated_fraction=vax[i]) for i in range(n_locations)]

    _, neighbors = get_neighbors_dict(locations)

    for ix, village in enumerate(villages):
        village.neighbors = [villages[v] for v in neighbors[ix]]

    return villages


if __name__ == '__main__':

    logging.basicConfig(format='%(message)s', level=logging.INFO)

    villages = initialize_topology(20)
    logging.debug(villages[1])

    villages[0].challenge(10)

    generation = 0
    while generation < 26:
        logging.info('\n------------------------------------------------------------'
                     '\ngeneration-%d'
                     '\n------------------------------------------------------------', generation)

        for v in villages:
            v.update()

        for v in villages:
            v.transmit()
            logging.info(v.summary())

        logging.info('------------------------------------------------------------')

        generation += 1
