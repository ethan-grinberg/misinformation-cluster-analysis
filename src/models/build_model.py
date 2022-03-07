# -*- coding: utf-8 -*-
import os
import logging
from pathlib import Path
from dotenv import find_dotenv, load_dotenv
import pandas as pd
from cluster_graphs import ClusterGraphs


def main(proj_dir):
    """ Runs data processing scripts to turn raw data from (../raw) into
        cleaned data ready to be analyzed (saved in ../processed).
    """
    logger = logging.getLogger(__name__)
    logger.info('extracting features from graphs')

    graph_data = pd.read_pickle(os.path.join(proj_dir, "data", "processed", "graph_data.pkl"))

    cluster = ClusterGraphs(graph_data)
    cluster.cluster_k_means(n_clusters=3)
    cluster_df = cluster.get_clustered_df()
    cluster_df.to_pickle(os.path.join(proj_dir, "models", "graphs_clustered.pkl"))

if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    project_dir = Path(__file__).resolve().parents[2]

    load_dotenv(find_dotenv())
    main(project_dir)
