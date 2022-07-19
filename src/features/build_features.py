# -*- coding: utf-8 -*-
import os
import logging
from pathlib import Path
from dotenv import find_dotenv, load_dotenv
import pandas as pd
from graph_embed import GraphEmbed
import sys


def main(proj_dir, pheme=False):
    """ Runs data processing scripts to turn raw data from (../raw) into
        cleaned data ready to be analyzed (saved in ../processed).
    """
    logger = logging.getLogger(__name__)
    logger.info('extracting features from graphs')

    if pheme == False:
        raw_data = pd.read_csv(os.path.join(proj_dir, "data", "raw", "all_networks.csv"))
        p_data_file = os.path.join(proj_dir, "data", "processed", "graph_data.pkl")
        data_model = "hoaxy"
    else:
        raw_data = pd.read_csv(os.path.join(proj_dir, "data", "raw", "pheme_data.csv"))
        p_data_file = os.path.join(proj_dir, "data", "processed", "graph_data_pheme.pkl")
        data_model = "pheme"

    GE = GraphEmbed(p_data_file, .15, 5, raw_data, data_model, emb_model="ugraphemb")
    embedding_df = GE.get_features()
    embedding_df.to_pickle(p_data_file)

if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    project_dir = Path(__file__).resolve().parents[2]
    
    load_dotenv(find_dotenv())
    main(project_dir)
