import pandas as pd
from data_model import DataModel

class HoaxyFeatures(DataModel):
    def __init__(self):
        pass
    def filter_data(self, raw_data, tolerance, min_edges):

        grouped_data = raw_data.groupby("id").filter(lambda x: len(x) >= min_edges)
        upper = grouped_data.id.value_counts().quantile(1-tolerance)
        lower = grouped_data.id.value_counts().quantile(tolerance)
        graph_df = grouped_data.groupby("id").filter(lambda x: (len(x) >= lower) & (len(x) <= upper))

        return graph_df
    def get_meta_data(self, raw_data, row_data, id):
        pass
    def get_network_data(self, graph, extra_data):
        pass
    def build_graphs(self):
        pass
