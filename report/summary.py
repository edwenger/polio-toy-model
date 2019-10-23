from .base import BaseConsoleReport, BaseFileReport


class SummaryReport(BaseFileReport):

    columns = ['generation', 'village', 'N', 'S', 'I']

    def __init__(self, directory='output', filename='summary.csv'):
        super().__init__(directory, filename)
        self.log.debug(','.join(self.columns))  # write header

    def village_summary(self, simulation, village):
            self.log.debug('%d,%d,%d,%d,%d',
                           simulation.generation, village.ix, village.N, village.S, village.I)


class ConsoleSummary(BaseConsoleReport):

    def __init__(self):
        super().__init__()

    def simulation_update_begin(self, simulation):
        self.log.debug('\n------------------------------------------------------------'
                       '\ngeneration-%d'
                       '\n------------------------------------------------------------',
                       simulation.generation)

    def village_summary(self, simulation, village):
        self.log.debug('%s', village.summary())

    def simulation_update_end(self):
        self.log.debug('------------------------------------------------------------')
