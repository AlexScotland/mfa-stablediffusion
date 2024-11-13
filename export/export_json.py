import json

from ..interfaces.iexporter import IExporter

class JSONExporter(IExporter):

    @staticmethod
    def export(item_to_export):
        with open(f"{item_to_export.directory}/config.json", 'w') as file:
            file.write(json.dumps(eval(repr(item_to_export))))