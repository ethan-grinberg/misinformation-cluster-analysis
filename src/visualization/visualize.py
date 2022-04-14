from json import tool
import altair as alt
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import networkx as nx
from sklearn.manifold import TSNE

class Visualize:
    def __init__(self, cluster_info, graphs=None):
        self.cluster_info = cluster_info
        self.graphs = graphs
        self.X = np.array(cluster_info.graph_embedding.to_list())

    def viz_graphs(self, ids):
        for i in range(len(ids)):
            plt.figure(i)
            nx.draw(self.graphs[ids[i]])
        plt.show()
    
    def get_graph(self, idx):
        id = self.cluster_info.iloc[idx].id
        return self.graphs[id]

    def graph_reduced_dimensions(self, tooltip_data):
        tsne = TSNE(2)
        two_d = tsne.fit_transform(self.X)

        components = pd.DataFrame(two_d, columns=['dim1', 'dim2'])
        components['label'] = self.cluster_info.label
        for col in tooltip_data:
            components[col] = self.cluster_info[col]
        
        chart = alt.Chart(components).mark_circle(size=60).encode(
                    x='dim1',
                    y='dim2',
                    color='label:N',
                    tooltip=tooltip_data
                ).interactive()

        return chart
    
    def plot_cluster_size(self, width=200, height=300):
        df  = pd.DataFrame(self.cluster_info.label.value_counts())
        df = df.rename({"label": "count"}, axis=1)
        df['label'] = df.index
        df = df.reset_index()

        chart = alt.Chart(df).mark_bar().encode(
            x='label:N',
            y='count',
            color="label:N"
        ).properties(width=width, height=height)

        return chart

    def graph_point_range_cluster_info(self, y_vals, width, height, cols):
        points = (
            alt.Chart()
            .mark_point(size=70)
            .transform_fold(fold=y_vals, as_=["category", "y"])
            .encode(x="label:N", y=alt.Y("mean(y):Q"), color="label:N")
        )

        point_range = (
            alt.Chart()
            .mark_errorbar(extent="ci")
            .transform_fold(fold=y_vals, as_=["category", "y"])
            .encode(x="label:N", y="mean(y):Q", color="label:N")
        )

        chart = (
            alt.layer(points, point_range, data=self.cluster_info)
            .properties(width=width, height=height)
            .facet(columns=cols, facet="category:N")
        )
        return chart
    
    def export_graphs_for_viz(self):
        cur = 0
        dfs = []
        for i in range(len(self.cluster_info)):
            edges = self.cluster_info.iloc[i].edges
            label = self.cluster_info.iloc[i].label
            arr = np.array(edges)
            for i in range(arr.max() + 1):
                arr = np.where(arr == i, cur, arr)
                cur += 1
            df = pd.DataFrame(arr, columns=['source', 'target'])
            df['label'] = label
            dfs.append(df)
        
        all_edges = pd.concat(dfs)
        return all_edges