import os
import logging


class TransmissionReport(object):

    def __init__(self):
        self.columns = ['generation', 'village', 'transmitter', 'infected']
        self.directory = 'output'

        self.log = logging.getLogger(__name__)
        self.log.setLevel(logging.DEBUG)

        # log configuration
        os.makedirs(self.directory, exist_ok=True)
        handler = logging.FileHandler("{0}/{1}.log".format(self.directory, 'transmission'), mode='w')
        handler.setFormatter(logging.Formatter('%(message)s'))
        handler.setLevel(logging.DEBUG)
        self.log.addHandler(handler)

        # write header
        self.log.debug(','.join(self.columns))

    def report(self, event, **kwargs):
        event_fn = getattr(self, event, None)
        if event_fn is not None:
            event_fn(**kwargs)

    def acquired_infection(self, simulation, village, transmitter, infected):

        self.log.debug('%d,%d,%s,%d',
                       simulation.generation, village.ix, transmitter.transmitter, infected.ix)
