import pandas as pd
import numpy as np
import networkx as nx
import karateclub.graph_embedding as ge
from u_graph_emb import UGraphEmb
import os
import EoN
# nlp libraries
import spacy
from spacy.language import Language
from spacy_langdetect import LanguageDetector
from spacytextblob.spacytextblob import SpacyTextBlob
from spacy.lang.en.stop_words import STOP_WORDS
from spacy.tokens import Doc

class GraphEmbed:
    @staticmethod
    def read_graphs(graph_data):
        graphs = [nx.DiGraph(e) for e in graph_data.edges]
        return {graph_data.iloc[i].id: graphs[i] for i in np.arange(len(graph_data))}
    
    # available graph embedding models
    models = {"feather": ge.FeatherGraph(), "graph2vec": ge.Graph2Vec(), "ugraphemb": UGraphEmb()}

    def __init__(self, 
                f_path,
                tolerance,
                min_edges,
                raw_networks,
                type=None, 
                model_params=None):

        # check whether or not embeddings were computed
        self.f_path = f_path
        if os.path.exists(f_path):
            self.has_embeddings = True
        else:
            self.has_embeddings = False

        # configure model parameters
        if not self.has_embeddings:
            self.model = self.models[type]
            if not model_params is None:
                for k, v in model_params.items():
                    self.model.__dict__[k] = v

        # filter out min number of edges from networks
        raw_networks = raw_networks.groupby("id").filter(lambda x: len(x) >= min_edges)
        upper = raw_networks.id.value_counts().quantile(1-tolerance)
        lower = raw_networks.id.value_counts().quantile(tolerance)
        self.graph_df = raw_networks.groupby("id").filter(lambda x: (len(x) >= lower) & (len(x) <= upper))
        
        # set the order of ids
        ids = list(self.graph_df.id.unique())
        ids.sort()
        self.ids = ids

        #load nlp library
        self.__load_spacy()

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
            self.__get_network_data(g, row_data, network_info)
            all_data.append(row_data)

            i += 1
        
        return pd.DataFrame(all_data)
    
    def __get_article_data(self, network_info, row_data, id):
        first_row = network_info.iloc[0]
        num_tweets = len(network_info)

        # retweet times
        times = network_info.tweet_created_at
        times  = pd.to_datetime(times).sort_values().to_list()
        time_r = (times[-1] - times[0]).total_seconds()

        # information types
        types = network_info.tweet_type.value_counts().to_dict()
        
        # article polarity and subjectivity
        pol, sub = self.__get_article_sent(first_row.title)

        # append data
        row_data['canonical_url'] = first_row.canonical_url
        row_data['date_published'] = first_row.date_published
        row_data['domain'] = first_row.domain
        row_data['id'] = id
        row_data['site_type'] = first_row.site_type
        row_data['title'] = first_row.title
        row_data['total_time'] = time_r
        row_data['retweet_num'] = types.get("retweet", 0) / num_tweets
        row_data['quote_num'] = types.get("quote", 0) / num_tweets
        row_data['reply_num'] = types.get("reply", 0) / num_tweets
        row_data['origin_num'] = types.get("origin", 0) / num_tweets
        row_data['article_lang'] = self.__get_article_lang(first_row.title)
        row_data['article_pol'] = pol
        row_data['article_subjectivity'] = sub

    def __get_network_data(self, graph, extra_data, network_info):
        # size
        edges = graph.edges
        nodes = graph.nodes
        num_nodes = len(nodes)
        num_edges = len(edges)

        # components
        wcc = [c for c in sorted(nx.weakly_connected_components(graph), key=len, reverse=True)]
        num_wcc = len(wcc)
        largest_wcc = len(wcc[0])

        # distance
        H = graph.subgraph(wcc[0])
        H = H.to_undirected()
        largest_di = nx.diameter(H)
        w_index = nx.wiener_index(H)

        #degree
        out_deg = [deg[1] for deg in graph.out_degree()]
        in_deg = [deg[1] for deg in graph.in_degree()]
        max_in = max(in_deg)
        max_out = max(out_deg)

        #epidemiology modeling
        R = EoN.estimate_R0(graph, transmissibility=.5)

        # append data
        extra_data['edges'] = list(edges)
        extra_data['num_nodes'] = num_nodes
        extra_data['num_edges'] = num_edges
        extra_data['num_wcc'] = num_wcc
        extra_data['largest_wcc'] = largest_wcc / num_nodes
        extra_data['diameter_largest_wcc'] = largest_di
        extra_data['max_out_degree'] = max_out
        extra_data['max_in_degree'] = max_in
        extra_data['mean_out_degree'] = sum(out_deg) / len(out_deg)
        extra_data['mean_in_degree'] = sum(in_deg) / len(in_deg)
        extra_data['average_time'] = extra_data['total_time'] / num_nodes
        extra_data['reproduction_num'] = R
        extra_data['wiener_index'] = w_index / num_nodes
    
    def __get_article_lang(self, title):
        if title is np.NaN:
            return np.NaN
        doc = self.nlp(title)
        return doc._.language['language']
    
    def __get_article_sent(self, title):
        if title is np.NaN:
            return np.NaN, np.NaN

        doc = self.nlp(title)
        pol = doc._.polarity
        sub = doc._.subjectivity
        return pol, sub

    def __get_lang_detector(self, nlp, name):
        return LanguageDetector()

    def __load_spacy(self):
        self.nlp = spacy.load("en_core_web_sm")
        Language.factory("language_detector", func=self.__get_lang_detector)
        self.nlp.add_pipe('language_detector', last=True)
        self.nlp.add_pipe("spacytextblob")
    
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