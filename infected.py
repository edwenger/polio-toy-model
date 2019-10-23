import itertools

import numpy as np


class Transmission(object):
    
    """ A container for info related to transmission event """

    def __init__(self, infection_id=None, village_id=None):
        self.infection_id = infection_id
        self.village_id = village_id

    @property
    def transmitter(self):
        return self.infection_id if self.infection_id is not None else ''


class Infected(object):

    """ An infected individual aware of its village context """

    uid = itertools.count()  # unique index generator

    internal_contact_rate = 5.0
    external_contact_rate = 2.0
    
    def __init__(self, village):
        self.ix = next(Infected.uid)
        self.village = village

    def transmit(self):
        """ Challenge local and neighboring villages with infectious contacts """
            
        transmitter_info = Transmission(self.ix, self.village.ix)

        n_internal_challenged = np.random.poisson(self.internal_contact_rate)
        self.village.challenge(n_internal_challenged, transmitter_info)
        
        n_external_challenged = np.random.poisson(self.external_contact_rate)
        
        if n_external_challenged == 0:
            return
        
        n_neighbors = len(self.village.neighbors)
        if n_neighbors > 0:
            neighbors_challenged = np.random.multinomial(n_external_challenged,
                                                         [1./n_neighbors]*n_neighbors)
                                                     
            for nv, nc in zip(self.village.neighbors, neighbors_challenged):
                if nc > 0:
                    nv.challenge(nc, transmitter_info)
