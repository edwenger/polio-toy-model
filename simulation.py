import logging


log = logging.getLogger(__name__)


class Params(object):

    def __init__(self, **kwargs):
        self.__dict__ = kwargs


class Simulation(object):

    """ A forward simulation of village states over a number of generations """

    def __init__(self, params):
        self.generation = 0
        initializer_fn, initializer_kwargs = params.pop('initializer_fn')
        self.villages = initializer_fn(self, **initializer_kwargs)
        self.reports = params.pop('reports', [])
        self.params = Params(**params)

    def run(self):
        for _ in range(self.params.sim_duration):
            self.update()

    def update(self):

        self.notify('simulation_generation_begin', simulation=self)

        for v in self.villages:
            v.update()

        for v in self.villages:
            v.transmit()
            self.notify('village_summary', simulation=self, village=v)

        self.notify('simulation_generation_end')

        self.generation += 1

    def notify(self, event, *args, **kwargs):
        for listener in self.reports:
            listener.report(event, *args, **kwargs)
