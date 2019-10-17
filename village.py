import itertools
import logging

import numpy as np

from infected import Infected


log = logging.getLogger(__name__)


class Village(object):

    """ A container for the population and immune state of a village """

    uid = itertools.count()  # unique index generator
    
    birth_rate = 0.03 / 26.  # per person fortnight
    
    def __init__(self, loc, N, vaccinated_fraction=0, neighbors=[]):
        self.ix = next(Village.uid)
        self.loc = tuple(loc)  # (x, y)
        self.N = N  # total population
        self.vaccinated_fraction = vaccinated_fraction
        self.S = int(N * (1.-vaccinated_fraction))  # initial state
        self.neighbors = neighbors  # [Village]
        self.previous_infecteds = []  # [Infected]
        self.infecteds = []  # [Infected]

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

    def challenge(self, n_transmissions, transmitter_info=None):
        """ Perform infectious contacts triggered by transmitter """

        infections = np.random.binomial(n_transmissions, float(self.S) / self.N)
        
        infections = min(self.S, infections)
        
        for _ in range(infections):
            self.S -= 1
            acquired_infection = Infected(village=self)
            logging.debug('\nAcquired infection-%d in village-%d',
                          acquired_infection.ix, self.ix)
                
            if transmitter_info is not None:
                logging.debug('==> transmitted by infection-%d in village-%d',
                              transmitter_info.infection_id, transmitter_info.village_id)

            self.infecteds.append(acquired_infection)

    @property
    def I(self):
        return len(self.infecteds)

    def summary(self):
        # return 'village-%d: S=%d I=%d N=%d' % (self.ix, self.S, self.I, self.N)
        return 'village-%d:\t(N,S,I) = \t%d\t%d\t%d' % (self.ix, self.N, self.S, self.I)
