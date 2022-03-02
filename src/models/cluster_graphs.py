import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from kneed import KneeLocator
from sklearn.metrics import pairwise_distances_argmin_min

class ClusterGraphs:
    def __init__(self, graph_vecs):
        self.graph_vecs = graph_vecs      

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

    def get_cluster_labels(self):
        return self.labels
    
    def get_mean_vec_cluster(self, cluster):
        idx = self.labels.loc[self.labels == cluster].index
        vecs = []
        for i in idx:
            vecs.append(self.graph_vecs[i])
        
        vecs = np.array(vecs)
        mean_vec = np.array([self.__get_mean_vec(vecs)])

        index = pairwise_distances_argmin_min(mean_vec, vecs)[0][0]
        return idx[index]
    
    def __get_mean_vec(self, vecs):
        return vecs.mean(axis=0)


    

