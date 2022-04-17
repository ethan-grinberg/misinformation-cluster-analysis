import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from kneed import KneeLocator
from sklearn.metrics import pairwise_distances_argmin_min

class ClusterGraphs:
    def __init__(self, graph_info_df):
        self.graph_info_df = graph_info_df.copy()
        self.graph_vecs = graph_info_df.graph_embedding.to_list()    

    def get_clustered_df(self):
        self.graph_info_df['label'] = self.labels.to_list()
        self.graph_info_df['is_mean_vec'] = np.repeat(False, len(self.graph_info_df))
        for i in self.get_closest_vec_centroid():
            self.graph_info_df.at[i, 'is_mean_vec'] = True
        
        return self.graph_info_df

    def choose_clust_num_k_means(self, end=11):
        inertias = []
        for i in range(1, end):
            kmeans = KMeans(n_clusters=i, random_state=0)
            kmeans.fit(self.graph_vecs)

            inertia = kmeans.inertia_
            inertias.append(inertia)

        # choose best cluster num
        kneedle = KneeLocator(
            range(1, end), inertias, S=1.0, curve="convex", direction="decreasing"
        )
        return kneedle.elbow, inertias
    
    def cluster_k_means(self, n_clusters=None):
        if n_clusters is None:
            n_clusters, inertias = self.choose_clust_num_k_means()
        
        kmeans = KMeans(n_clusters=n_clusters, random_state=0)
        kmeans.fit(self.graph_vecs)

        self.labels = pd.Series(kmeans.labels_)
        self.centers = kmeans.cluster_centers_

    def get_cluster_labels(self):
        return self.labels
    
    def get_closest_vec_centroid(self):
       closest, _ = pairwise_distances_argmin_min(self.centers, self.graph_vecs)
       return closest
    
    def __get_mean_vec(self, vecs):
        return vecs.mean(axis=0)


    

