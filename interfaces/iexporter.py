from abc import ABC, abstractmethod

class IExporter(ABC):

    @abstractmethod
    def export(item_to_export):
        pass