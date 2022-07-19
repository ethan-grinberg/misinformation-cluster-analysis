from matplotlib.pyplot import title
import pandas as pd
import numpy as np
import networkx as nx
import karateclub.graph_embedding as ge
from u_graph_emb import UGraphEmb
import os
from hoaxy_features import HoaxyFeatures
from pheme_features import PhemeFeatures

class GraphEmbed:
    @staticmethod
    def read_graphs(graph_data):
        graphs = [nx.DiGraph(e) for e in graph_data.edges]
        return {graph_data.iloc[i].id: graphs[i] for i in np.arange(len(graph_data))}
    
    # available graph embedding models
    models = {"feather": ge.FeatherGraph(), "graph2vec": ge.Graph2Vec(), "ugraphemb": UGraphEmb()}

    def __init__(self, 
                processed_data,
                tolerance,
                min_edges,
                raw_data,
                is_pheme,
                type=None, 
                model_params=None):
        
        self.is_pheme = is_pheme

        # check whether or not embeddings were computed
        self.processed_data = processed_data
        if os.path.exists(processed_data):
            self.has_embeddings = True
        else:
            self.has_embeddings = False

        # configure model parameters
        if not self.has_embeddings:
            self.model = self.models[type]
            if not model_params is None:
                for k, v in model_params.items():
                    self.model.__dict__[k] = v
        
        if self.is_pheme:
            self.data_model = PhemeFeatures()
        else:
            self.data_model = HoaxyFeatures()


        # filter out min number of edges from networks
        self.graph_df = self.data_model.filter_data(raw_data, tolerance, min_edges)
        
        # set the order of ids
        ids = list(self.graph_df.id.unique())
        ids.sort()
        self.ids = ids

    def get_features(self):
        df = pd.DataFrame()

        # get all networks as nx objects
        self.graphs = self.data_model.build_graphs(self.ids, self.graph_df)

        if not self.has_embeddings:
            self.__fit()
            embeddings = self.__get_embedding().tolist()
            df['graph_embedding'] = embeddings
        else:
            df = pd.read_pickle(self.processed_data)
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

        for id in self.ids:
            g = self.graphs[id]
            network_info = self.graph_df.loc[self.graph_df.id == id]

            row_data = {}
            self.data_model.get_meta_data(network_info, row_data, id)
            self.data_model.get_network_data(g, row_data)
            all_data.append(row_data)
        
        return pd.DataFrame(all_data)        

    def __fit(self):
        self.model.fit(self.graphs)
    
    def __get_embedding(self):
        return self.model.get_embedding()