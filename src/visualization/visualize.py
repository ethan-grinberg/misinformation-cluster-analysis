from typing_extensions import final
import altair as alt
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import networkx as nx
from sklearn.manifold import TSNE
import seaborn as sns
import os
from networkx.drawing.nx_pydot import graphviz_layout

class Visualize:
    @staticmethod
    def get_super(x):
        normal = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+-=()"
        super_s = "ᴬᴮᶜᴰᴱᶠᴳᴴᴵᴶᴷᴸᴹᴺᴼᴾQᴿˢᵀᵁⱽᵂˣʸᶻᵃᵇᶜᵈᵉᶠᵍʰᶦʲᵏˡᵐⁿᵒᵖ۹ʳˢᵗᵘᵛʷˣʸᶻ⁰¹²³⁴⁵⁶⁷⁸⁹⁺⁻⁼⁽⁾"
        res = x.maketrans(''.join(normal), ''.join(super_s))
        return x.translate(res)
    
    @staticmethod
    def wrap_by_word(s, n):
        a = s.split()
        ret = ''
        for i in range(0, len(a), n):
            ret += ' '.join(a[i:i+n]) + '\n'

        return ret

    def __init__(self, cluster_info, graphs=None):
        self.cluster_info = cluster_info
        self.graphs = graphs

        self.X = {}
        for k, v in self.cluster_info.items():
            self.X[k] = np.array(v.graph_embedding.to_list())
    
    # Currently not working properly
    def viz_graphs(self, ids, save=False, data_dir=None):
        for k,v in self.cluster_info.items():

            viz = v.loc[v.id.isin(ids[k])].copy()
            viz.loc[:, 'title'] = viz.loc[:, 'title'].fillna("no title")

            labels = viz.label.to_list() 
            cur_ids = viz.id.to_list()
            titles = viz.title.to_list()

            i = 0
            for id in cur_ids:
                g = self.graphs[k][id]
                pos = graphviz_layout(g, prog="dot")

                t = self.wrap_by_word(titles[i], 8)
                t = t + '\n ' + k

                plt.figure(i, figsize=(5,5))
                ax = plt.gca()
                ax.set_title(t, fontsize='x-large', fontweight='bold')

                if labels[i] == 0:
                    nx.draw(g, node_color="#1f77b4", ax=ax, pos=pos)
                elif labels[i] == 1:
                    nx.draw(g, node_color="#ff7f0e", ax=ax, pos=pos)
                else:
                    nx.draw(g, node_color="#d62728", ax=ax, pos=pos)

                i += 1

            if save:
                plt.savefig(os.path.join(data_dir, 'central_networks_' + str(k) + '.svg'))

        if not save:
            plt.show()
    
    def __get_graph_layout(self, g):
        df = pd.DataFrame(index=g.nodes(), columns=g.nodes())
        for row, data in nx.shortest_path_length(g):
            for col, dist in data.items():
                df.loc[row,col] = dist

        df = df.fillna(df.max().max())

        layout = nx.kamada_kawai_layout(g, dist=df.to_dict())
        return layout

    def viz_type_clusters(self, variable):
        df = self.cluster_info.copy()
        if variable == "article_lang":
            df = df.loc[df.article_lang != "en"].copy()
        chart =alt.Chart(df).mark_bar(
            cornerRadiusTopLeft=3,
            cornerRadiusTopRight=3
        ).encode(
            x=alt.X(variable+":N", sort='-y'),
            y=alt.Y('count():Q'),
            color='label:N'
        )
        return chart

    def get_graph(self, idx):
        id = self.cluster_info.iloc[idx].id
        return self.graphs[id]

    def graph_reduced_dimensions(self, tooltip_data, width, height, title):
        tsne = TSNE(2)
        two_d = tsne.fit_transform(self.X)

        components = pd.DataFrame(two_d, columns=['dimension 1', 'dimension 2'])
        components['label'] = self.cluster_info.label
        for col in tooltip_data:
            components[col] = self.cluster_info[col]
        
        chart = alt.Chart(components).mark_circle(size=60).encode(
                    x='dimension 1',
                    y='dimension 2',
                    color='label:N',
                    tooltip=tooltip_data
                ).properties(title=title, width=width, height=height).interactive()
        
        chart = chart.configure_title(fontSize=25, fontWeight='bold')
        chart = chart.configure_header(titleFontSize=25, titleFontWeight='bold')
        chart = chart.configure_legend(titleFontSize=25, labelFontSize=20, labelFontWeight='bold', titleFontWeight='bold')
        chart = chart.configure_axis(grid=False, titleFontSize=20, labelFontSize=15, labelAngle=0)

        return chart
    
    def plot_cluster_size(self, width=200, height=300):
        charts = []
        for k, v in self.cluster_info.items():
            df  = pd.DataFrame(v.label.value_counts())
            df = df.rename({"label": "count"}, axis=1)
            df['label'] = df.index
            df = df.reset_index()

            chart = alt.Chart(df).mark_bar().encode(
                x='label:N',
                y='count',
                color="label:N"
            ).properties(width=width, height=height, title=k)

            charts.append(chart)

        return alt.hconcat(*charts)

    def graph_point_range_cluster_info(self, rename, y_vals, width, height, cols):
        charts = []
        if rename:
            col_mapper = y_vals
            y_vals = list(y_vals.values())
        for k,v in self.cluster_info.items():
            clust = v.copy()
            if rename:
                clust.rename(columns=col_mapper, inplace=True)

            points = (
                alt.Chart()
                .mark_circle(size=200)
                .transform_fold(fold=y_vals, as_=[k, "y"])
                .encode(x="label:N", y=alt.Y("mean(y):Q"), color="label:N")
            )

            point_range = (
                alt.Chart()
                .mark_errorbar(extent="ci")
                .transform_fold(fold=y_vals, as_=[k, "y"])
                .encode(x="label:N", y=alt.Y("mean(y):Q"), color="label:N", strokeWidth=alt.value(2))
            )

            chart = (
                alt.layer(points, point_range, data=clust)
                .properties(title=k, width=width, height=height)
                .facet(columns=cols, facet=k+":N")
            )

            charts.append(chart)

        final_chart =  alt.hconcat(*charts)
        return final_chart
    
    def __configure_alt_chart(self, chart):
        chart = chart.configure_header(title=None, labelFontSize=25, labelFontWeight='bold')
        chart = chart.configure_legend(titleFontSize=25, labelFontSize=20, labelFontWeight='bold', titleFontWeight='bold')
        chart = chart.configure_axis(title=None, labelFontSize=15, labelAngle=0)
        return chart
    
    def get_corr_heat_map(self, features):
        plt.figure(figsize=(16, 6))
        corr = self.cluster_info[features].corr()
        return sns.heatmap(corr, xticklabels=corr.columns, yticklabels=corr.columns, vmin=-1, vmax=1, annot=True)
    
    def export_graphs_for_viz(self):
        cur = 0
        dfs = []
        for i in range(len(self.cluster_info)):
            edges = self.cluster_info.iloc[i].edges
            label = self.cluster_info.iloc[i].label
            mean = self.cluster_info.iloc[i].is_mean_vec
            arr = np.array(edges)
            for i in range(arr.max() + 1):
                arr = np.where(arr == i, cur, arr)
                cur += 1
            df = pd.DataFrame(arr, columns=['source', 'target'])
            df['label'] = label
            df['is_mean_vec'] = mean
            dfs.append(df)
        
        all_edges = pd.concat(dfs)
        central_edges = all_edges.loc[all_edges.is_mean_vec == True]
        return all_edges, central_edges