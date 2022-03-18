import pandas as pd
import numpy as np
import networkx as nx
import karateclub.graph_embedding as ge
import os

class GraphEmbed:
    models = {"feather": ge.FeatherGraph(), "graph2vec": ge.Graph2Vec()}

    def __init__(self, type=None, params=None):
        if type is None:
            self.embed = False
        else:
            self.embed = True
            self.model = self.models[type]
            if not params is None:
                for k, v in params.items():
                    self.model.__dict__[k] = v
    
    def build_graphs(self, graph_df, min_edges):
        self.graphs = []
        self.data = []
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
                self.graphs.append(relabel_g)
                self.ids.append(id)

                # append data
                extra_data = {}
                extra_data['canonical_url'] = network.iloc[0].canonical_url
                extra_data['date_published'] = network.iloc[0].date_published
                extra_data['domain'] = network.iloc[0].domain
                extra_data['id'] = id
                extra_data['site_type'] = network.iloc[0].site_type
                extra_data['title'] = network.iloc[0].title
                extra_data['edges'] = list(relabel_g.edges)

                self.__get_network_data(relabel_g, extra_data)
                self.data.append(extra_data)

    def __get_network_data(self, graph, extra_data):
        extra_data['num_nodes'] = len(graph.nodes)
        extra_data['num_edges'] = len(graph.edges)
        extra_data['num_strongly_connected'] = nx.number_strongly_connected_components(graph)
        extra_data['num_weakly_connected'] = nx.number_weakly_connected_components(graph)
        extra_data['average_clustering_coef'] = nx.average_clustering(graph)
    
    def read_graphs(self, graph_data):
        graphs = [nx.DiGraph(e) for e in graph_data.edges]
        return graphs

    def get_graphs(self):
        return self.graphs

    def fit(self):
        self.model.fit(self.graphs)
    
    def get_embedding(self):
        return self.model.get_embedding()
    
    def get_embedding_df(self):
        df = pd.DataFrame(self.data)
        df['graph_embedding'] = self.get_embedding().tolist()
        return df


