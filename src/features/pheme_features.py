import pandas as pd
from data_model import DataModel
from functools import reduce
import numpy as np
import networkx as nx


class PhemeFeatures(DataModel):
    def __init__(self):
        pass

    def filter_data(self, raw_data, tolerance, min_edges):
        #rename thread to be consistent in code
        grouped_data = raw_data.rename(columns={'thread': 'id'})

        # truth value of non-rumors is non-rumor
        grouped_data['truth'] = grouped_data['truth'].fillna('non-rumor')

        grouped_data = grouped_data.groupby("id").filter(lambda x: len(x) >= min_edges)
        upper = grouped_data.id.value_counts().quantile(1-tolerance)
        lower = grouped_data.id.value_counts().quantile(tolerance)
        graph_df = grouped_data.groupby("id").filter(lambda x: (len(x) >= lower) & (len(x) <= upper))
        
        return graph_df.reset_index(drop=True)

    def get_meta_data(self, raw_data, row_data, id):
        first_row = raw_data.iloc[0]

        row_data['id'] = id
        row_data['truth'] = first_row.truth
        row_data['title'] = first_row.title
        row_data['event'] = first_row.event
    
    def get_network_data(self, graph, extra_data):
        edges = graph.edges
        nodes = graph.nodes
        num_nodes = len(nodes)
        num_edges = len(edges)

        # append data
        extra_data['edges'] = list(edges)
        extra_data['num_nodes'] = num_nodes
        extra_data['num_edges'] = num_edges

    def build_graphs(self, ids, graph_df):
        graphs = {}
        for id in ids:
            net = graph_df.loc[(graph_df.id == id) & (graph_df.is_source_tweet != 1)]
            g = self._build_graph(net)
            graphs[id] = g
        return graphs
    
    def _build_graph(self, net):
        g = nx.from_pandas_edgelist(net, "user_id", "in_reply_user", create_using=nx.DiGraph)
        relabled_g = nx.convert_node_labels_to_integers(g)

        # add out degree centrality attribute to nodes
        centrality = nx.out_degree_centrality(relabled_g)
        centrality = {key:[value] for (key,value) in centrality.items()}
        nx.set_node_attributes(relabled_g, centrality, "out-degree")

        return relabled_g

    def agg_tweets_by_thread(df):
    
        shared = lambda x: 1 - len(set(x)) / len(x)
        shared.__name__ = "shared"

        funcs = [np.mean, sum, np.var]
        agg_props = {
            "favorite_count": funcs,
            "user_mentions": funcs,
            "media_count": funcs,
            "sensitive": funcs,
            "has_place": funcs,
            "has_coords": funcs,
            "retweet_count": funcs,
            "hashtags_count": funcs + [shared],
            "urls_count": funcs,
            "user.tweets_count": funcs,
            "is_rumor": max,
            "tweet_id": len,
            "user.has_bg_img": funcs,
            "has_quest": funcs,
            "has_exclaim": funcs,
            "has_quest_or_exclaim": funcs,
            "user.default_pic": funcs,
            "has_smile_emoji": funcs,
            "user.verified": funcs,
            "user.name_length": funcs,
            "user.handle_length": funcs,
            "user.profile_sbcolor": funcs,
            "user.profile_bgcolor": funcs,
            
            "hasperiod": funcs,
            "number_punct": funcs,
            "negativewordcount" : funcs,
            "positivewordcount" : funcs,
            "capitalratio" : funcs,
            "contentlength" : funcs,
            "sentimentscore" : funcs,
            "Noun" : funcs,
            "Verb" : funcs,
            "Adjective" : funcs,
            "Pronoun" : funcs,
            "Adverb": funcs,
        }
        rename = {
            "tweet_id": "thread_length"
        }
        
        # Step 1: Build simple aggregate features
        agg = df.groupby("thread")\
            .agg(agg_props)\
            .rename(columns=rename)
        
        agg.columns = [ "_".join(x) for x in agg.columns.ravel() ]
        agg = agg.rename(columns={"is_rumor_max": "is_rumor", "thread_length_len": "thread_length"})
        
        # Step 2: Builds some features off the source tweet, which has tweet_id == thread            
        src = df[df["is_source_tweet"] == 1][["thread",
                                            "user.followers_count", 
                                            "user.listed_count",
                                            "user.verified",
                                            "created",
                                            "user.created_at",
                                            "user.tweets_count"]] \
                            .rename(columns={"user.followers_count": "src.followers_count",
                                            "user.listed_count": "src.listed_count",
                                            "user.verified": "src.user_verified",
                                            "user.created_at": "src.created_at",
                                            "user.tweets_count": "src.tweets_total"})
        
        # Step 3: Build features off of the reply tweets
        def f(x):
            d = []
            
            # Get various features from the distribution of times of reply tweet
            d.append(min(x["created"]))
            d.append(max(x["created"]))
            d.append(np.var(x["created"]))
                    
            return pd.Series(d, index=["first_resp", "last_resp","resp_var"])
            
        replies = df[df["is_source_tweet"] == False] \
            .groupby("thread") \
            .apply(f)
        
        dfs = [agg, src, replies]
        thrd_data = reduce(lambda left, right: pd.merge(left,right, on="thread"), dfs)
        
        # Step 3: Add miscelaneous features
        # Remember timestamps increase as time progresses
        # src.created_at < created < first_resp < last_resp
        thrd_data["time_to_first_resp"] = thrd_data["first_resp"] - thrd_data["created"]
        thrd_data["time_to_last_resp"] = thrd_data["last_resp"] - thrd_data["created"]
        
        return thrd_data