import pandas as pd
import numpy as np
import os
from .visualize import Visualize
import sys
import altair as alt
from altair_saver import save
alt.renderers.enable('altair_saver', fmts=['vega-lite', 'png', 'svg'])

sys.path.append("..")
from features.build_features import GraphEmbed

def export_analysis(project_dir):
    export_folder = os.path.join(project_dir, 'reports', 'figures', 'pheme')
    cluster_f_name = 'pheme_graphs_clustered.pkl'
    archived_models = os.path.join(project_dir, "data", "archived")
    default_a_type = 'user_level'

    all_clusters = {}
    all_graphs = {}
    for f in os.listdir(archived_models):
        if f.startswith("."):
            continue
        clusters = pd.read_pickle(os.path.join(archived_models, f, cluster_f_name))
        clusters['analysis_type'] = f
        all_clusters[f] = clusters

        graphs = GraphEmbed.read_graphs(clusters)
        all_graphs[f] = graphs

    clusters = pd.read_pickle(os.path.join(project_dir, "models", cluster_f_name))
    clusters['analysis_type'] = default_a_type
    all_clusters[default_a_type] = clusters

    graphs = GraphEmbed.read_graphs(clusters)
    all_graphs[default_a_type] = graphs

    analysis_types = list(all_clusters.keys())
    viz = Visualize(all_clusters, all_graphs)

    # export central networks
    # central_ids = {}
    # for k, v in all_clusters.items():
    #     central_ids[k] = v.loc[v.is_mean_vec == True].id.to_list()

    # viz.viz_graphs(central_ids, save=True, data_dir=os.path.join(export_folder))

    # export numerical point range plots
    num_features = clusters.select_dtypes(include=['int16', 'int32', 'int64', 'float16', 'float32', 'float64']).columns.to_list()
    for feature in num_features:
        chart = viz.graph_point_range_cluster_info(False, [feature], 200, 250, 2)
        save(chart, os.path.join(export_folder, feature + '.svg'))