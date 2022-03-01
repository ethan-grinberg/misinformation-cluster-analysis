import pandas as pd
import numpy as np
import networkx as nx
import karateclub.graph_embedding as ge

class GraphEmbed:
    models = {"feather": ge.FeatherGraph(), "graph2vec": ge.Graph2Vec()}
    def __init__(self, type, params=None):
        self.model = self.models[type]
        if not params is None:
            for k, v in params.items():
                self.model.__dict__[k] = v
    
    def build_graphs(self, graph_df, min_edges):
        self.graphs = []
        self.ids = []
        for id in graph_df.id.unique():
            network = graph_df.loc[graph_df.id == id]
            if (len(network) < min_edges):
                continue
            else:
                edges = list(zip(network.from_user_id, network.to_user_id))
                g = nx.DiGraph()
                g.add_edges_from(edges)
                relabel_g = nx.convert_node_labels_to_integers(g)
                # append data
                self.graphs.append(relabel_g)
                self.ids.append(id)

    def get_graphs(self):
        return self.graphs

    def fit(self):
        self.model.fit(self.graphs)
    
    def get_embedding(self):
        return self.model.get_embedding()
    
    def get_embedding_df(self):
        data = list(zip(self.ids, self.get_embedding()))
        return pd.DataFrame(data, columns=["article_id", "graph_embedding"])


