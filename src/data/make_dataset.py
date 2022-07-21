# -*- coding: utf-8 -*-
import os
import logging
from pathlib import Path
from dotenv import find_dotenv, load_dotenv
from collect_hoaxy import HoaxyApi
from collect_pheme import collect_tweets_thread_data
import pandas as pd
import sys


# either make the dataset from the hoaxy api or scrape from eu vs disinfo and twitter
def main(proj_dir, hoaxy=False):
    """ Runs data processing scripts to turn raw data from (../raw) into
        cleaned data ready to be analyzed (saved in ../processed).
    """
    logger = logging.getLogger(__name__)
    logger.info('creating final dataset')

    ext_data = os.path.join(proj_dir, "data", "external")
    raw_data = os.path.join(proj_dir, "data", "raw")

    if hoaxy == True:
        # iffy list that hoaxy uses
        iffy_list = pd.read_csv(os.path.join(ext_data, "iffy+.csv"))
        domains = iffy_list.Domain.to_list()

        # get networks for all domains
        h_api = HoaxyApi(os.environ.get("RAPID"))
        networks = h_api.get_networks_all_domains(domains)
        networks.to_csv(os.path.join(raw_data, "all_networks.csv"), index=False)
    else:
        data_dir = os.path.join(ext_data, "pheme-rnr-dataset")
        output_dir = os.path.join(raw_data, "pheme")
        collect_tweets_thread_data(data_dir, output_dir, "pheme_all_events")

if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    project_dir = Path(__file__).resolve().parents[2]

    load_dotenv(find_dotenv())
    main(project_dir)
