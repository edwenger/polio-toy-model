import itertools
import logging

import numpy as np

from infected import Infected, Transmission


log = logging.getLogger(__name__)


class Village(object):

    """ A container for the population and immune state of a village """

    uid = itertools.count()  # unique index generator
    
    birth_rate = 0.03 / 26.  # per person fortnight
    
    def __init__(self, loc, N, vaccinated_fraction=0, neighbors=[], sim=None):
        self.ix = next(Village.uid)
        self.loc = tuple(loc)  # (x, y)
        self.N = N  # total population
        self.vaccinated_fraction = vaccinated_fraction
        self.S = int(N * (1.-vaccinated_fraction))  # initial state
        self.neighbors = neighbors  # [Village]
        self.previous_infecteds = []  # [Infected]
        self.infecteds = []  # [Infected]
        self.sim = sim  # simulation context

    def __str__(self):
        return '%s(id=%d, loc=%s, N=%d, vaccinated_fraction=%0.2f, neighbor_ids=%s)' \
               % (self.__class__.__name__,
                  self.ix, self.loc, self.N, self.vaccinated_fraction,
                  {v.ix for v in self.neighbors})

    def update(self):
        
        """ Move infecteds from past timestep """
        self.previous_infecteds = list(self.infecteds)
        self.infecteds = []

        """ Update susceptible population to account for vaccination-adjusted birth + death """
        expected_unvaccinated_birth = self.birth_rate * (1.-self.vaccinated_fraction) * self.N
        expected_susceptible_death = self.birth_rate * self.S

        self.S += np.random.poisson(expected_unvaccinated_birth) - np.random.poisson(expected_susceptible_death)

        self.S = min(max(self.S, 0), self.N)
    
    def transmit(self):
        """ Transmit infections from last infecteds to next infecteds """

        for infected in self.previous_infecteds:
            infected.transmit()

    def challenge(self, n_transmissions, transmitter_info=Transmission()):
        """ Perform infectious contacts triggered by transmitter """

        infections = np.random.binomial(n_transmissions, float(self.S) / self.N)
        
        infections = min(self.S, infections)
        
        for _ in range(infections):
            self.S -= 1
            acquired_infection = Infected(village=self)

            self.sim.notify('acquired_infection',
                            simulation=self.sim,
                            village=self,
                            transmitter=transmitter_info,
                            infected=acquired_infection
                            )

            self.infecteds.append(acquired_infection)

    @property
    def I(self):
        return len(self.infecteds)

    def summary(self):
        return 'village-%d:\t(N,S,I) = \t%d\t%d\t%d' % (self.ix, self.N, self.S, self.I)
