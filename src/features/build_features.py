# -*- coding: utf-8 -*-
import os
from pathlib import Path
from dotenv import find_dotenv, load_dotenv
import pandas as pd
from .graph_embed import GraphEmbed
import sys


def build_features(proj_dir, pheme=True, tweet_level=False, unverified_tweets=True, group_by_title=False, filter=True):
    """ Runs data processing scripts to turn raw data from (../raw) into
        cleaned data ready to be analyzed (saved in ../processed).
    """
    print('building features')

    if pheme == False:
        raw_data = pd.read_csv(os.path.join(proj_dir, "data", "raw", "all_networks.csv"))
        p_data_file = os.path.join(proj_dir, "data", "processed", "graph_data.pkl")
        data_model = "hoaxy"
    else:
        raw_data = pd.read_csv(os.path.join(proj_dir, "data", "raw", "pheme", "pheme_all_events.csv"))
        p_data_file = os.path.join(proj_dir, "data", "processed", "graph_data_pheme.pkl")
        data_model = "pheme"

    GE = GraphEmbed(p_data_file, 
        .15, 
        5, 
        raw_data, 
        data_model, 
        emb_model="ugraphemb", 
        tweet_level=tweet_level, 
        unverified_tweets=unverified_tweets,
        group_by_title=group_by_title,
        filter=filter)
    
    embedding_df = GE.get_features()
    embedding_df.to_pickle(p_data_file)
