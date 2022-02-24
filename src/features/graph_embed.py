import pandas as pd
import numpy as np
import networkx as nx

class GraphEmbed:
    def __init__(self):
        pass

    def build_graphs(self, graph_df, min_edges):
        graphs = []
        for id in graph_df.id.unique():
            network = graph_df.loc[graph_df.id == id]
            if (len(network) < min_edges):
                continue
            else:
                edges = list(zip(network.from_user_id, network.to_user_id))
                g = nx.DiGraph()
                g.add_edges_from(edges)
                graphs.append(g)

        return graphs

