import logging

import numpy as np

from triangulate import get_neighbors_dict
from village import Village
from simulation import Simulation

from report.summary import SummaryReport, ConsoleSummary
from report.transmission import TransmissionReport
from report.initialization import InitializationReport


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


def initialize_topology(sim, n_locations=500):

    locations = np.random.rand(n_locations, 2)
    _, neighbors = get_neighbors_dict(locations)
    N = PowerLawPopulationDistribution()(np.random.rand(n_locations))
    vax = np.random.uniform(0.25, 0.75, size=n_locations)

    villages = [Village(loc=locations[i], N=N[i], vaccinated_fraction=vax[i], sim=sim) for i in range(n_locations)]

    _, neighbors = get_neighbors_dict(locations)

    for ix, village in enumerate(villages):
        village.neighbors = [villages[v] for v in neighbors[ix]]

    return villages


if __name__ == '__main__':

    log_formatter = logging.Formatter('%(message)s')

    root_log = logging.getLogger()
    root_log.setLevel(logging.INFO)

    ch = logging.StreamHandler()
    ch.setFormatter(log_formatter)
    ch.setLevel(logging.INFO)
    root_log.addHandler(ch)

    params = dict(sim_duration=26,
                  reports=[
                      SummaryReport(),
                      ConsoleSummary(),
                      TransmissionReport(),
                      InitializationReport()
                  ],
                  initializer_fn=[
                      initialize_topology,
                      dict(n_locations=20)
                  ])

    sim = Simulation(params)
    logging.debug(sim.villages[1])

    sim.villages[0].challenge(10)

    sim.run()
