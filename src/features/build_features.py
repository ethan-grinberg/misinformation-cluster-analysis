# -*- coding: utf-8 -*-
import os
import logging
from pathlib import Path
from dotenv import find_dotenv, load_dotenv
import pandas as pd
from graph_embed import GraphEmbed


def main(proj_dir):
    """ Runs data processing scripts to turn raw data from (../raw) into
        cleaned data ready to be analyzed (saved in ../processed).
    """
    logger = logging.getLogger(__name__)
    logger.info('extracting features from graphs')

    all_networks = pd.read_csv(os.path.join(proj_dir, "data", "raw", "all_networks.csv"))

    GE = GraphEmbed(type="feather")
    GE.build_graphs(all_networks, 5)
    GE.fit()
    embedding_df = GE.get_embedding_df()
    embedding_df.to_pickle(os.path.join(proj_dir, "data", "processed", "graph_data.pkl"))
    GE.write_graphs(os.path.join(proj_dir, "data", "processed", "graphs"))

if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    project_dir = Path(__file__).resolve().parents[2]

    load_dotenv(find_dotenv())
    main(project_dir)
