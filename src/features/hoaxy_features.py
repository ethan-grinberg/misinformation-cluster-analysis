import pandas as pd
import networkx as nx
import numpy as np
from data_model import DataModel
import EoN

# nlp modules
import spacy
from spacy.language import Language
from spacy_langdetect import LanguageDetector
from spacytextblob.spacytextblob import SpacyTextBlob
from spacy.lang.en.stop_words import STOP_WORDS
from spacy.tokens import Doc

class HoaxyFeatures(DataModel):
    def __init__(self):
        #load nlp library
        self.__load_spacy()

    def filter_data(self, raw_data, tolerance, min_edges):

        grouped_data = raw_data.groupby("id").filter(lambda x: len(x) >= min_edges)
        upper = grouped_data.id.value_counts().quantile(1-tolerance)
        lower = grouped_data.id.value_counts().quantile(tolerance)
        graph_df = grouped_data.groupby("id").filter(lambda x: (len(x) >= lower) & (len(x) <= upper))

        return graph_df

    def get_meta_data(self, raw_data, row_data, id):
        first_row = raw_data.iloc[0]
        num_tweets = len(raw_data)

        # retweet times
        times = raw_data.tweet_created_at
        times  = pd.to_datetime(times).sort_values().to_list()
        time_r = (times[-1] - times[0]).total_seconds()

        # information types
        types = raw_data.tweet_type.value_counts().to_dict()
        
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

    def get_network_data(self, graph, extra_data):
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

    def build_graphs(self, ids, graph_df):
        graphs = {}
        for id in ids:
            network = graph_df.loc[graph_df.id == id]
            
            # append to list of graphs
            g = self.__build_graph(network)
            graphs[id] = g
        return graphs

    def __build_graph(self, network):
        edges = list(zip(network.from_user_id, network.to_user_id))
        g = nx.DiGraph()
        g.add_edges_from(edges)
        relabel_g = nx.convert_node_labels_to_integers(g)

        # add out degree centrality attribute to nodes
        centrality = nx.out_degree_centrality(relabel_g)
        centrality = {key:[value] for (key,value) in centrality.items()}
        nx.set_node_attributes(relabel_g, centrality, "out-degree")

        return relabel_g
    
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
