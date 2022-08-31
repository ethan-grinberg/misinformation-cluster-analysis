from sklearn.preprocessing import StandardScaler

class CustomEmbed:
    FEATURES = ['time_per_node', 
            'nodes_per_thread', 
            'largest_wcc', 
            'diameter_largest_wcc', 
            'num_nodes', 
            'sentiment_mean', 
            'user_tweet_count_mean', 
            'urls_mean', 
            'media_count_mean', 
            'user_follower_count_mean', 
            'user_friends_count_mean', 
            'num_edges', 
            'unverified']

    def __init__(self):
        pass
    
    def fit(self, extra_features):
        scalar = StandardScaler()
        X = extra_features[:, self.FEATURES]
        self.X = scalar.fit_transform(X)

    def get_embedding(self):
        return self.X
