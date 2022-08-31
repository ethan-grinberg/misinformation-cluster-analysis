# -*- coding: utf-8 -*-
import os
import logging
from pathlib import Path
from dotenv import find_dotenv, load_dotenv
import pandas as pd
from .cluster_graphs import ClusterGraphs
import sys
sys.path.append("..")
from visualization.visualize import Visualize


def build_model(proj_dir, vis=False, pheme=True, n_clusters=None):
    """ Runs data processing scripts to turn raw data from (../raw) into
        cleaned data ready to be analyzed (saved in ../processed).
    """
    print("building model")
    
    if pheme:
        graph_data = pd.read_pickle(os.path.join(proj_dir, "data", "processed", "graph_data_pheme.pkl"))
        out_file = os.path.join(proj_dir, "models", "pheme_graphs_clustered.pkl")
    else:
        graph_data = pd.read_pickle(os.path.join(proj_dir, "data", "processed", "graph_data.pkl"))
        out_file = os.path.join(proj_dir, "models", "graphs_clustered.pkl")

    cluster = ClusterGraphs(graph_data)
    cluster.cluster_k_means(n_clusters=n_clusters)
    cluster_df = cluster.get_clustered_df()
    cluster_df.to_pickle(out_file)

    if vis:
        # export graphs with labelled clusters to file for visualization
        viz = Visualize(cluster_df)
        graphs, central = viz.export_graphs_for_viz()
        graphs.to_csv(os.path.join(proj_dir, "reports", "graph_edges.csv"))
        central.to_csv(os.path.join(proj_dir, "reports", "central_edges.csv"))