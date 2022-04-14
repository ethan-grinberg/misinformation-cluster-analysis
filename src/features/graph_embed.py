import pandas as pd
import numpy as np
import networkx as nx
import karateclub.graph_embedding as ge
from u_graph_emb import UGraphEmb
import os

class GraphEmbed:
    models = {"feather": ge.FeatherGraph(), "graph2vec": ge.Graph2Vec(), "ugraphemb": UGraphEmb()}
    
    @staticmethod
    def read_graphs(graph_data):
        graphs = [nx.DiGraph(e) for e in graph_data.edges]
        return {graph_data.iloc[i].id: graphs[i] for i in np.arange(len(graph_data))}

    def __init__(self, 
                f_path,
                min_edges,
                raw_networks,
                type=None, 
                model_params=None):

        self.f_path = f_path
        if os.path.exists(f_path):
            self.has_embeddings = True
        else:
            self.has_embeddings = False

        if not self.has_embeddings:
            self.model = self.models[type]
            if not model_params is None:
                for k, v in model_params.items():
                    self.model.__dict__[k] = v

        # filter out min number of edges from networks
        self.min_edges = min_edges
        self.graph_df = raw_networks.groupby("id").filter(lambda x: len(x) >= self.min_edges)
        
        # set the order of ids
        ids = list(self.graph_df.id.unique())
        ids.sort()
        self.ids = ids

    def get_features(self):
        df = pd.DataFrame()
        self.__build_graphs()
        if not self.has_embeddings:
            self.__fit()
            embeddings = self.__get_embedding().tolist()
            df['graph_embedding'] = embeddings
        else:
            df = pd.read_pickle(self.f_path)
            old_ids = df.id.to_list()
            df = pd.DataFrame(df.graph_embedding)

        extra_features = self.__get_extra_features()

        # make sure the precomputed graph embeddings 
        # line up with new data
        if self.has_embeddings:
            if not list(extra_features.id) == old_ids:
                raise ValueError("pre computed graph embeddings don't line up with extra features")
            
        combined_df = pd.concat([df, extra_features], axis=1)
        return combined_df
    
    def __get_extra_features(self):
        all_data = []
        i = 0
        for id in self.ids:
            g = self.graphs[i]
            network_info = self.graph_df.loc[self.graph_df.id == id]

            row_data = {}
            row_data['canonical_url'] = network_info.iloc[0].canonical_url
            row_data['date_published'] = network_info.iloc[0].date_published
            row_data['domain'] = network_info.iloc[0].domain
            row_data['id'] = id
            row_data['site_type'] = network_info.iloc[0].site_type
            row_data['title'] = network_info.iloc[0].title
            row_data['edges'] = list(g.edges)

            self.__get_network_data(g, row_data)
            
            all_data.append(row_data)
            i += 1
        
        return pd.DataFrame(all_data) 

    def __build_graphs(self):
        self.graphs = []
        for id in self.ids:
            network = self.graph_df.loc[self.graph_df.id == id]
            
            # append to list of graphs
            g = self.__create_graph(network)
            self.graphs.append(g)

    def __create_graph(self, network):
        edges = list(zip(network.from_user_id, network.to_user_id))
        g = nx.DiGraph()
        g.add_edges_from(edges)
        relabel_g = nx.convert_node_labels_to_integers(g)

        # add out degree centrality attribute to nodes
        centrality = nx.out_degree_centrality(relabel_g)
        centrality = {key:[value] for (key,value) in centrality.items()}
        nx.set_node_attributes(relabel_g, centrality, "out-degree")

        return relabel_g

    def __get_network_data(self, graph, extra_data):
        extra_data['num_nodes'] = len(graph.nodes)
        extra_data['num_edges'] = len(graph.edges)
        extra_data['num_strongly_connected'] = nx.number_strongly_connected_components(graph)
        extra_data['num_weakly_connected'] = nx.number_weakly_connected_components(graph)

    def __fit(self):
        self.model.fit(self.graphs)
    
    def __get_embedding(self):
        return self.model.get_embedding()