from abc import ABC, abstractmethod

from plotly.graph_objs import Figure
class FairplotCreator(ABC):
    @abstractmethod
    def factory_method(self):
        pass

    def abstract_figure_creation(self):
        product = self.factory_method()
        figure = product.get_figure()
        return figure