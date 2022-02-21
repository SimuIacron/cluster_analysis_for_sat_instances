import os
from matplotlib import pyplot as plt
from DataFormats.DbInstance import DbInstance
from UtilScripts import util


# plots the distributions of base and/or gate features for the given cluster, can exclude selected features
def boxplot_cluster_feature_distribution(cluster, db_instance: DbInstance, dpi=192, use_base=False, use_gate=False,
                                         angle=90, output_file='', show_plot=False, exclude_features=None):
    if exclude_features is None:
        exclude_features = []
    plot_data = []
    x_labels = []

    if use_base:
        for feature_name, feature_data in zip(db_instance.base_f, util.rotateNestedLists(cluster['base_01'])):
            if feature_name not in exclude_features:
                plot_data.append(feature_data)
                x_labels.append(feature_name)
    if use_gate:
        for feature_name, feature_data in zip(db_instance.base_f, util.rotateNestedLists(cluster['gate_01'])):
            if feature_name not in exclude_features:
                plot_data.append(feature_data)
                x_labels.append(feature_name)

    fig = plt.figure(figsize=(1700 / dpi, 1000 / dpi), dpi=dpi)
    ax = fig.add_subplot(111)
    ax.boxplot(plot_data)
    plt.xticks(range(1, len(x_labels) + 1), x_labels, rotation=angle)
    plt.tight_layout()

    if output_file != '':
        fig.savefig(os.environ['TEXPATH'] + output_file + '.svg')

    if show_plot:
        fig.show()