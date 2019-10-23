import os
import logging


class BaseReport(object):

    def __init__(self):

        self.log = logging.getLogger('_'.join([__name__, self.__class__.__name__]))
        self.log.setLevel(logging.DEBUG)

    def report(self, event, **kwargs):

        event_fn = getattr(self, event, None)

        if event_fn is not None:
            event_fn(**kwargs)


class BaseFileReport(BaseReport):

    def __init__(self, directory, filename='base.log'):

        super().__init__()

        self.directory = directory
        os.makedirs(self.directory, exist_ok=True)

        self.filename = filename
        handler = logging.FileHandler("{0}/{1}".format(self.directory, self.filename), mode='w')

        handler.setFormatter(logging.Formatter('%(message)s'))
        handler.setLevel(logging.DEBUG)
        self.log.addHandler(handler)


class BaseConsoleReport(BaseReport):

    def __init__(self):

        super().__init__()

        handler = logging.StreamHandler()

        handler.setFormatter(logging.Formatter('%(message)s'))
        handler.setLevel(logging.DEBUG)
        self.log.addHandler(handler)
