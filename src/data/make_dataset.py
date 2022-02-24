# -*- coding: utf-8 -*-
import os
import logging
from pathlib import Path
from dotenv import find_dotenv, load_dotenv
from collect_eu_disinfo import EuDisinfo
from twitter_scrape import TwitterScraper
from collect_hoaxy import HoaxyApi
import pandas as pd


# either make the dataset from the hoaxy api or scrape from eu vs disinfo and twitter
def main(proj_dir, hoaxy=True):
    """ Runs data processing scripts to turn raw data from (../raw) into
        cleaned data ready to be analyzed (saved in ../processed).
    """
    logger = logging.getLogger(__name__)
    logger.info('creating final dataset')

    ext_data = os.path.join(proj_dir, "data", "external")
    raw_data = os.path.join(proj_dir, "data", "raw")
    if hoaxy:
        # iffy list that hoaxy uses
        iffy_list = pd.read_csv(os.path.join(ext_data, "iffy+.csv"))
        domains = iffy_list.Domain.to_list()

        # get networks for all domains
        h_api = HoaxyApi(os.environ.get("RAPID"))
        networks = h_api.get_networks_all_domains(domains)
        networks.to_csv(os.path.join(raw_data, "all_networks.csv"), index=False)
    else:
        combined_file = os.path.join(ext_data, "combined_claims.pkl")

        # combine eu data from downloaded source
        disinfo = EuDisinfo(os.environ.get("DRIVER_PATH"))
        combined_data = disinfo.get_downloaded_dataset(ext_data)

        # export combined dataset
        combined_data.to_pickle(combined_file)

        # find all tweets for each claim if it contains a link to the article
        claim_df = pd.read_pickle(combined_file)
        t_scrape = TwitterScraper()
        all_tweets = t_scrape.find_tweets_all_articles(claim_df)
        
        # export tweet
        all_tweets.to_csv(os.path.join(raw_data, "all_article_tweets.csv"),index=False)

if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    project_dir = Path(__file__).resolve().parents[2]

    load_dotenv(find_dotenv())
    main(project_dir)
