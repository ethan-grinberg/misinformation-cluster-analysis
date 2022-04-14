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
            self.__get_article_data(network_info, row_data, id)
            self.__get_network_data(g, row_data)
            all_data.append(row_data)

            i += 1
        
        return pd.DataFrame(all_data)
    
    def __get_article_data(self, network_info, row_data, id):
        row_data['canonical_url'] = network_info.iloc[0].canonical_url
        row_data['date_published'] = network_info.iloc[0].date_published
        row_data['domain'] = network_info.iloc[0].domain
        row_data['id'] = id
        row_data['site_type'] = network_info.iloc[0].site_type
        row_data['title'] = network_info.iloc[0].title

    def __get_network_data(self, graph, extra_data):
        # size
        edges = graph.edges
        nodes = graph.nodes
        num_nodes = len(nodes)
        num_edges = len(edges)

        # components
        scc = list(nx.strongly_connected_components(graph))
        wcc = [c for c in sorted(nx.weakly_connected_components(graph), key=len, reverse=True)]
        
        num_scc = len(scc)
        num_wcc = len(wcc)
        largest_scc = max([len(comp) for comp in scc])
        largest_wcc = len(wcc[0])

        # distance
        H = graph.subgraph(wcc[0])
        H = H.to_undirected()
        largest_di = nx.diameter(H)

        #degree
        out_deg = [deg[1] for deg in graph.out_degree()]
        in_deg = [deg[1] for deg in graph.in_degree()]
        max_in = max(in_deg)
        max_out = max(out_deg)

        # append data
        extra_data['edges'] = list(edges)
        extra_data['num_nodes'] = num_nodes
        extra_data['num_edges'] = num_edges
        extra_data['num_scc'] = num_scc / num_nodes
        extra_data['num_wcc'] = num_wcc / num_nodes
        extra_data['largest_scc'] = largest_scc / num_nodes
        extra_data['largest_wcc'] = largest_wcc / num_nodes
        extra_data['diameter_largest_wcc'] = largest_di / num_nodes
        extra_data['max_out_degree'] = max_out / num_nodes
        extra_data['max_in_degree'] = max_in / num_nodes
    
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

    def __fit(self):
        self.model.fit(self.graphs)
    
    def __get_embedding(self):
        return self.model.get_embedding()