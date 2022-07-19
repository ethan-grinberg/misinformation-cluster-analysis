import pandas as pd
from abc import ABC, abstractmethod

class DataModel(ABC):
    @abstractmethod
    def filter_data(self, raw_data, tolerance, min_edges):
        pass

    @abstractmethod
    def get_meta_data(self, raw_data, row_data, id):
        pass

    @abstractmethod
    def get_network_data(self, graph, extra_data):
        pass
    
    @abstractmethod
    def build_graphs(self, ids, graph_df):
        pass
