from data.make_dataset import make_dataset
from features.build_features import build_features
from models.build_model import build_model
from visualization.export_analysis import export_analysis

from dotenv import find_dotenv, load_dotenv
import os
from pathlib import Path

def main(project_dir):
    # data pipeline settings
    pipeline = {'make_dataset': False, 'build_features': False, 'build_model': False, 'export_analysis': True}
    make_dataset_args = {'hoaxy': False}
    build_features_args = {'pheme': True, 'tweet_level': False, 'unverified_tweets': True}
    build_model_args = {'vis': False, "pheme": True}
    export_analysis_args = {}

    # run the data pipeline
    for i in range(len(pipeline)):
        if pipeline['make_dataset']:
            make_dataset(project_dir, **make_dataset_args)
            pipeline['make_dataset'] = False
        if pipeline['build_features']:
            build_features(project_dir, **build_features_args)
            pipeline['build_features'] = False
        if pipeline['build_model']:
            build_model(project_dir, **build_model_args)
            pipeline['build_model'] = False
        if pipeline['export_analysis']:
            export_analysis(project_dir, **export_analysis_args)
            pipeline['export_analysis'] = False

if __name__ == '__main__':
    project_dir = Path(__file__).resolve().parents[1]
    load_dotenv(find_dotenv())
    main(project_dir)