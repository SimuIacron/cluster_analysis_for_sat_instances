import os

import numpy as np
from matplotlib import pyplot as plt

from DataFormats.DbInstance import DbInstance
from util_scripts.util import random_color


def plot_family_distribution_of_clusters(data_clusters, db_instance: DbInstance, output_file='',
                                         show_plot=False, dpi=192):
    biggest_family_size = []
    biggest_family_name = []
    other_families_size = []
    percentage_familes = []

    for cluster in data_clusters:
        biggest_family_name.append(cluster['family_list'][0][0])
        biggest_family_size.append(cluster['family_list'][0][1])
        percentage_familes.append(cluster['family_total_percentage'])
        size = 0
        for family in cluster['family_list'][1:]:
            size = size + family[1]
        other_families_size.append(size)

    X_axis = range(len(data_clusters))

    plt.figure(figsize=(1200 / dpi, 1000 / dpi), dpi=dpi)
    plt.barh(X_axis, biggest_family_size, label='Biggest Family')
    plt.barh(X_axis, other_families_size, label='Other Families', left=biggest_family_size)

    for x, family in zip(X_axis, biggest_family_name):
        plt.annotate('{family} ({size}/{total_size}), {percentage}% of family'.format(family=family, size=biggest_family_size[x],
                                                             total_size=biggest_family_size[x] + other_families_size[
                                                                 x], percentage=round(percentage_familes[x] * 100)),
                     (150, x+0.1), rotation=0)

    plt.legend(loc='upper right')
    plt.ylabel('Cluster')
    plt.xlabel('Size')
    plt.tight_layout()
    plt.yticks(X_axis)
    plt.gca().invert_yaxis()

    if output_file != '':
        plt.savefig(os.environ['TEXPATH'] + output_file + '.svg')

    if show_plot:
        plt.show()


def plot_runtime_comparison_sbs(data_cluster, sbs_solver, output_file='', show_plot=False, dpi=192):
    runtimes_sbs = []
    runtimes_cluster_solver = []
    solvers = [cluster['cluster_par2'][0][0][0] for cluster in data_cluster]
    for cluster in data_cluster:
        # cluster_solver = cluster['cluster_par2']
        runtimes_cluster_solver.append(cluster['cluster_par2'][0][0][1])

        for solver_tuple in cluster['cluster_par2'][0]:
            if solver_tuple[0] == sbs_solver:
                runtimes_sbs.append(solver_tuple[1])

    X_axis = range(len(data_cluster))

    plt.figure(figsize=(1200 / dpi, 1000 / dpi), dpi=dpi)

    plt.scatter(runtimes_sbs, X_axis, zorder=2, label='SBS')
    plt.barh(X_axis, runtimes_cluster_solver, zorder=1, label='CSBS')

    plt.ylabel('Cluster')
    plt.xlabel('Par2-Score (s)')

    for x, solver in zip(X_axis, solvers):
        plt.annotate(solver, (1000, x+0.1), rotation=0)

    plt.legend(loc='upper right')
    plt.tight_layout()
    plt.yticks(X_axis)
    plt.gca().invert_yaxis()

    if output_file != '':
        plt.savefig(os.environ['TEXPATH'] + output_file + '.svg')

    if show_plot:
        plt.show()


def plot_performance_mean_std_and_performance_of_cluster(data_clusters, db_instance: DbInstance, output_file='',
                                                         show_plot=False, dpi=192):
    mean_list = []
    std_list = []
    solvers = [cluster['cluster_performance_solver'] for cluster in data_clusters]

    for cluster in data_clusters:
        bps = cluster['cluster_performance_solver']
        bps_index = db_instance.solver_f.index(bps)
        mean_list.append(cluster['runtimes_mean'][bps_index])
        std_list.append(cluster['runtimes_std'][bps_index])

    X_axis = range(len(data_clusters))

    plt.figure(figsize=(1200 / dpi, 600 / dpi), dpi=dpi)

    plt.bar(range(len(data_clusters)), mean_list, label='Mean')
    plt.bar(range(len(data_clusters)), std_list, label='Standard Deviation', bottom=mean_list)

    plt.xlabel('Cluster')
    plt.ylabel('Seconds (s)')

    for x, solver in zip(X_axis, solvers):
        plt.annotate(solver, (x+0.1, 1000), rotation=90)

    plt.legend()
    plt.tight_layout()

    if output_file != '':
        plt.savefig(os.environ['TEXPATH'] + output_file + '.svg')

    if show_plot:
        plt.show()


def boxplot_runtimes_distribution_per_cluster(data_clusters, db_instance: DbInstance, output_file='',
                                              show_plot=False, dpi=192):
    boxplot_list = []
    solvers = [cluster['cluster_performance_solver'] for cluster in data_clusters]

    for cluster in data_clusters:
        bps = cluster['cluster_performance_solver']
        bps_index = db_instance.solver_f.index(bps)
        boxplot_list.append([item[bps_index] for item in cluster['runtimes']])

    X_axis = range(len(data_clusters))

    plt.figure(figsize=(1200 / dpi, 600 / dpi), dpi=dpi)

    plt.boxplot(boxplot_list, labels=X_axis)

    plt.xlabel('Cluster')
    plt.ylabel('Seconds (s)')

    for x, solver in zip(X_axis, solvers):
        plt.annotate(solver, (x + 1-0.05, 1000), rotation=90)

    # plt.legend()
    plt.tight_layout()

    if output_file != '':
        plt.savefig(os.environ['TEXPATH'] + output_file + '.svg')

    if show_plot:
        plt.show()


def plot_biggest_family_runtime(data_clusters, output_file='', show_plot=False, dpi=192):
    par2_scores_cluster = []
    par2_scores_family_in_cluster = []
    par2_scores_total_family = []

    biggest_family_size = []
    biggest_family_name = []
    other_families_size = []
    solver_name = []

    for cluster in data_clusters:
        par2_scores_total_family.append(cluster['complete_family_par2'])
        par2_scores_cluster.append(cluster['cluster_performance_solver_par2'])
        par2_scores_family_in_cluster.append(cluster['family_in_cluster_par2'])
        biggest_family_name.append(cluster['family_list'][0][0])
        biggest_family_size.append(cluster['family_list'][0][1])
        solver_name.append(cluster['cluster_performance_solver'])
        size = 0
        for family in cluster['family_list'][1:]:
            size = size + family[1]
        other_families_size.append(size)

    X_axis = np.arange(len(data_clusters))

    plt.figure(figsize=(1200 / dpi, 600 / dpi), dpi=dpi)
    plt.xlabel('Cluster')
    plt.ylabel('Par2 (s)')
    bar_size = 0.2
    plt.bar(X_axis - bar_size, par2_scores_family_in_cluster, bar_size, label='Family in Cluster')
    plt.bar(X_axis, par2_scores_cluster, bar_size, label='Complete Cluster')
    plt.bar(X_axis + bar_size, par2_scores_total_family, bar_size, label='Complete Family')

    for x, family in zip(X_axis, biggest_family_name):
        plt.annotate('{family} ({size}/{total_size}), {solver}'.format(family=family, size=biggest_family_size[x],
                                                                            total_size=biggest_family_size[x] +
                                                                                       other_families_size[x],
                                                                            solver=solver_name[x]),
                     (x-0.05, 250), rotation=90)

    plt.tight_layout()
    plt.legend()

    if output_file != '':
        plt.savefig(os.environ['TEXPATH'] + output_file + '.svg')

    if show_plot:
        plt.show()
