# -*- coding: utf-8 -*-
import os
from pathlib import Path
from dotenv import find_dotenv, load_dotenv
from .collect_hoaxy import HoaxyApi
from .collect_pheme import collect_all_events
import pandas as pd
import sys


# either make the dataset from the hoaxy api or scrape from eu vs disinfo and twitter
def make_dataset(proj_dir, hoaxy=False):
    """ Runs data processing scripts to turn raw data from (../raw) into
        cleaned data ready to be analyzed (saved in ../processed).
    """
    print("making dataset")

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
        collect_all_events(data_dir, output_dir, "pheme_all_events")
