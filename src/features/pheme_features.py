import pandas as pd
from .data_model import DataModel
from functools import reduce
import numpy as np
import networkx as nx


class PhemeFeatures(DataModel):
    def __init__(self, tweet_level=False, unverified_tweets=True, group_by_title=False, filter=True):
        self.tweet_level = tweet_level
        self.unverified_tweets = unverified_tweets
        self.group_by_title = group_by_title
        self.filter = filter

    def filter_data(self, raw_data, tolerance, min_edges):
        #rename thread to be consistent in code
        if self.group_by_title:
           grouped_data =  self._group_by_title(raw_data)
        else:
            grouped_data = raw_data.rename(columns={'thread': 'id'})

        # 0 ids have no reply
        grouped_data['in_reply_user'] = grouped_data['in_reply_user'].replace(0, pd.NA)
        grouped_data['in_reply_tweet'] = grouped_data['in_reply_tweet'].replace(0, pd.NA)

        # truth value of non-rumors is non-rumor
        grouped_data = grouped_data.loc[grouped_data.truth.notnull()].copy()

        if not self.unverified_tweets:
            grouped_data = grouped_data.loc[~(grouped_data.truth == 'unverified')].copy()
        
        # make sure graphs meet minimum size
        grouped_data = grouped_data.groupby("id").filter(lambda x: len(x) >= min_edges)

        if self.filter:
            upper = grouped_data.id.value_counts().quantile(1-tolerance)
            lower = grouped_data.id.value_counts().quantile(tolerance)
            graph_df = grouped_data.groupby("id").filter(lambda x: (len(x) >= lower) & (len(x) <= upper))
        
            return graph_df.reset_index(drop=True)
        else:
            return grouped_data.reset_index(drop=True)

    def get_meta_data(self, raw_data, row_data, id):
        first_row = raw_data.iloc[0]

        # time data
        times = pd.to_datetime(raw_data.created * 1e6)
        times = times.sort_values().to_list()
        total_time = (times[-1] - times[0]).total_seconds()

        row_data['id'] = id
        row_data['truth'] = first_row.truth
        row_data['title'] = first_row.title
        row_data['event'] = first_row.event
        row_data['mean_tweet_len'] = raw_data.tweet_length.mean()
        row_data['mean_user_mentions'] = raw_data.user_mentions.mean()
        row_data['urls_mean'] = raw_data.urls_count.mean()
        row_data['media_count_mean'] = raw_data.media_count.mean()
        row_data['hashtags_count_mean'] = raw_data.hashtags_count.mean()
        row_data['retweet_count_mean'] = raw_data.retweet_count.mean()
        row_data['favorite_count_mean'] = raw_data.favorite_count.mean()
        row_data['mentions_count_mean'] = raw_data.mentions_count.mean()
        row_data['user_tweet_count_mean'] = raw_data['user.tweets_count'].mean()
        row_data['user_follower_count_mean'] = raw_data['user.followers_count'].mean()
        row_data['user_friends_count_mean'] = raw_data['user.friends_count'].mean()
        row_data['sentiment_mean'] = raw_data['sentimentscore'].mean()
        row_data['total_time'] = total_time

        # encoding truth values
        if first_row.truth == 'unverified':
            row_data['unverified'] = 1
        else:
            row_data['unverified'] = 0

        if first_row.truth == 'true':
            row_data['truth_val'] = 1
        else:
            row_data['truth_val'] = 0

        # number of threads
        if self.group_by_title:
            row_data['num_threads'] = raw_data.thread.nunique()
    
    def get_network_data(self, graph, extra_data):
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
        extra_data['wiener_index'] = w_index / num_nodes
        extra_data['time_per_node'] = extra_data['total_time'] / num_nodes

        if self.group_by_title:
            extra_data['nodes_per_thread'] = num_nodes  / extra_data['num_threads']

    def build_graphs(self, ids, graph_df):
        graphs = {}
        for id in ids:
            net = graph_df.loc[graph_df.id == id]
            g = self._build_graph(net)
            graphs[id] = g
        return graphs
    
    def _build_graph(self, net):
        net = net.dropna(subset=['in_reply_tweet', 'in_reply_user'])
        if self.tweet_level:
            g = nx.from_pandas_edgelist(net, "in_reply_tweet", "tweet_id", create_using=nx.DiGraph)
        else:
            g = nx.from_pandas_edgelist(net, "in_reply_user", "user_id", create_using=nx.DiGraph)

        relabled_g = nx.convert_node_labels_to_integers(g)

        # add out degree centrality attribute to nodes
        centrality = nx.out_degree_centrality(relabled_g)
        centrality = {key:[value] for (key,value) in centrality.items()}
        nx.set_node_attributes(relabled_g, centrality, "out-degree")

        return relabled_g
    
    def _group_by_title(self, raw_data):
        grouped_data = raw_data.copy()
        grouped_data['id'] = 0
        i = 0
        for title in grouped_data.loc[grouped_data.title != "none"].title.unique():
            grouped_data.loc[grouped_data.title == title, 'id'] = i
            i+=1
        return grouped_data

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
            "retweet_count": funcs,
            "hashtags_count": funcs + [shared],
            "urls_count": funcs,
            "user.verified": funcs
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