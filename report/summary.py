import os
import logging


class SummaryReport(object):

    def __init__(self):
        self.columns = ['generation', 'village', 'N', 'S', 'I']
        self.directory = 'output'

        self.log = logging.getLogger(__name__)
        self.log.setLevel(logging.DEBUG)

        # log configuration
        os.makedirs(self.directory, exist_ok=True)
        handler = logging.FileHandler("{0}/{1}.log".format(self.directory, 'summary'), mode='w')
        handler.setFormatter(logging.Formatter('%(message)s'))
        handler.setLevel(logging.DEBUG)
        self.log.addHandler(handler)

        # write header
        self.log.debug(','.join(self.columns))

    def report(self, event, **kwargs):
        event_fn = getattr(self, event, None)
        if event_fn is not None:
            event_fn(**kwargs)

    def village_summary(self, simulation, village):
            self.log.debug('%d,%d,%d,%d,%d',
                           simulation.generation, village.ix, village.N, village.S, village.I)


class ConsoleSummary(object):

    def __init__(self):

        self.log = logging.getLogger('ConsoleSummary')
        self.log.setLevel(logging.DEBUG)

        # log configuration
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter('%(message)s'))
        handler.setLevel(logging.DEBUG)
        self.log.addHandler(handler)

    def report(self, event, **kwargs):
        event_fn = getattr(self, event, None)
        if event_fn is not None:
            event_fn(**kwargs)

    def simulation_generation_begin(self, simulation):
        self.log.debug('\n------------------------------------------------------------'
                       '\ngeneration-%d'
                       '\n------------------------------------------------------------',
                       simulation.generation)

    def village_summary(self, simulation, village):
        self.log.debug('%s', village.summary())

    def simulation_generation_end(self):
        self.log.debug('------------------------------------------------------------')
