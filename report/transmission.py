from .base import BaseFileReport


class TransmissionReport(BaseFileReport):

    columns = ['generation', 'village', 'transmitter', 'infected']

    def __init__(self, directory='output', filename='transmission.csv'):
        super().__init__(directory, filename)
        self.log.debug(','.join(self.columns))  # write header

    def acquired_infection(self, simulation, village, transmitter, infected):
        self.log.debug('%d,%d,%s,%d',
                       simulation.generation, village.ix, transmitter.transmitter, infected.ix)
