import yaml

from ..interfaces.iexporter import IExporter

class YAMLExporter(IExporter):

    @staticmethod
    def export(item_to_export):
        with open(f"{item_to_export.directory}/config.yml", 'w') as file:
            file.write(yaml.dump(item_to_export.to_dict()))