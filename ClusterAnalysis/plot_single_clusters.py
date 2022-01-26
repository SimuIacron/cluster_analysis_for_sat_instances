import os

import numpy as np
from matplotlib import pyplot as plt

from DataFormats.DbInstance import DbInstance
from util_scripts import util


def boxplot_cluster_feature_distribution(cluster, db_instance: DbInstance, dpi=192, use_base=False, use_gate=False,
                                         angle=90, output_file='', show_plot=False):
    plot_data = []
    x_labels = []

    if use_base:
        plot_data = plot_data + util.rotateNestedLists(cluster['base_01'])
        x_labels = x_labels + db_instance.base_f
    if use_gate:
        plot_data = plot_data + util.rotateNestedLists(cluster['gate_01'])
        x_labels = x_labels + db_instance.gate_f

    fig = plt.figure(figsize=(1700 / dpi, 1000 / dpi), dpi=dpi)
    ax = fig.add_subplot(111)
    ax.boxplot(plot_data)
    plt.xticks(range(1, len(x_labels) + 1), x_labels, rotation=angle)
    plt.tight_layout()

    if output_file != '':
        fig.savefig(os.environ['TEXPATH'] + output_file + '.svg')

    if show_plot:
        fig.show()


def barchart_compare_sbs_speedup(cluster_list, dpi=192, output_file='', show_plot=False):
    X_labels = generate_family_bar_plot_labels(cluster_list)
    sbs_speedup = [cluster['sbs_deviation_factor'] for cluster in cluster_list]
    solvers = [cluster['cluster_deviation_solver'] for cluster in cluster_list]

    X_axis = np.arange(len(X_labels))

    fig, ax1 = plt.subplots(figsize=(1700 / dpi, 1000 / dpi), dpi=dpi)
    ax1.set_xlabel('Families')
    plt.xticks(X_axis, X_labels, rotation=90)

    bar_size = 0.3
    ax1.bar(X_axis, sbs_speedup, bar_size, label='SBS Speedup', color='tab:green')

    for x, solver in zip(X_axis, solvers):
        ax1.annotate(solver, (x, max(sbs_speedup) - 2), rotation=90)

    plt.legend()

    plt.tight_layout()

    if output_file != '':
        fig.savefig(os.environ['TEXPATH'] + output_file + '.svg')

    if show_plot:
        fig.show()


def barchart_compare_runtime_scores(cluster_list, db_instance: DbInstance, dpi=192, output_file='', show_plot=False):

    X_labels = generate_family_bar_plot_labels(cluster_list)

    # X_labels = [cluster['family_list'][0][0] for cluster in cluster_list]
    solvers = [cluster['cluster_deviation_solver'] for cluster in cluster_list]
    cluster_par2_csbs = [(cluster['cluster_par2'][0][0][1]) for cluster in cluster_list]

    cluster_par2_bps = []
    for cluster in cluster_list:
        for solver_tuple in cluster['cluster_par2'][0]:
            if solver_tuple[0] == cluster['cluster_deviation_solver']:
                cluster_par2_bps.append(solver_tuple[1])
                break

    cluster_deviation_score = [cluster['cluster_deviation_score'] for cluster in cluster_list]
    complete_family_par2_bps = [cluster['complete_family_par2'] for cluster in cluster_list]

    X_axis = np.arange(len(X_labels))

    fig, ax1 = plt.subplots(figsize=(1700 / dpi, 1000 / dpi), dpi=dpi)
    ax1.set_xlabel('Families')
    plt.xticks(X_axis, X_labels, rotation=90)

    bar_size = 0.2
    ax1.bar(X_axis - bar_size * 1.5, cluster_par2_csbs, bar_size, label='Cluster Par2 (CSBS)', color='tab:red')
    ax1.bar(X_axis - bar_size * 0.5, cluster_par2_bps, bar_size, label='Cluster Par2 (BRS)', color='tab:green')
    ax1.bar(X_axis + bar_size * 0.5, cluster_deviation_score, bar_size, label='Cluster Robustness Score', color='tab:orange')
    ax1.bar(X_axis + bar_size * 1.5, complete_family_par2_bps, bar_size, label='Complete Family Par2 (BPS)', color='tab:blue')
    ax1.set_ylim((0, 11000))

    for x, solver in zip(X_axis, solvers):
        ax1.annotate(solver, (x, 8000), rotation=90)

    plt.legend()

    plt.tight_layout()

    if output_file != '':
        fig.savefig(os.environ['TEXPATH'] + output_file + '.svg')

    if show_plot:
        fig.show()


# -- Helper functions --------------------------------------------------------------------------------------------------

def generate_family_bar_plot_labels(cluster_list):
    X_labels = []
    for cluster in cluster_list:
        text = ''
        for family in cluster['family_list'][:1]:
            text = text + family[0] + '\n'
        X_labels.append(text)

    return X_labels