import json

from .base import BaseFileReport


class InitializationReport(BaseFileReport):

    def __init__(self, directory='output', filename='initialization.json'):
        super().__init__(directory, filename)

    def villages_init(self, villages):
        self.log.debug(json.dumps([v.to_dict() for v in villages]))
