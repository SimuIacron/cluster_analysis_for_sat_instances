import os

import numpy as np
from matplotlib import pyplot as plt

from DataFormats.DbInstance import DbInstance
from util_scripts.util import random_color


def plot_family_distribution_of_clusters(data_clusters, db_instance: DbInstance, biggest_n_families=2, output_file='',
                                         show_plot=False, dpi=192):

    other_label = 'other'
    colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'tab:orange', 'navy', 'tab:gray', 'gold', 'deeppink']

    family_list = []
    families = list(set([family[0] for family in db_instance.family_wh]))
    families.append(other_label)
    data_clusters_edited = []

    for cluster in data_clusters:
        cluster_edited = cluster['family_list'][:biggest_n_families]
        count = 0
        for tuple_ in cluster['family_list'][biggest_n_families:]:
            count = count + tuple_[1]
        cluster_edited.append((other_label, count))
        data_clusters_edited.append(cluster_edited)

    for family in families:
        current_family_list = []
        for cluster in data_clusters_edited:
            found_tuple = False
            for tuple_ in cluster:
                if tuple_[0] == family:
                    current_family_list.append(tuple_[1])
                    found_tuple = True
                    break

            if not found_tuple:
                current_family_list.append(0)

        family_list.append(current_family_list)

    families_edited = []
    family_list_edited = []
    for i, family in enumerate(families):
        hasNonZeroEntry = False
        for entry in family_list[i]:
            if entry != 0:
                hasNonZeroEntry = True
                break
        if hasNonZeroEntry:
            families_edited.append(family)
            family_list_edited.append(family_list[i])

    families = families_edited
    family_list = family_list_edited

    plt.figure(figsize=(1200 / dpi, 600 / dpi), dpi=dpi)

    for i in range(len(family_list)):
        bottom = [0] * len(data_clusters)
        for j in range(i):
            bottom = np.array(bottom) + np.array(family_list[j])
        plt.bar(range(len(data_clusters)), family_list[i], label=families[i], bottom=bottom,
                color=colors[i % len(colors)])

    plt.legend(ncol=2)
    plt.xlabel('Cluster')
    plt.ylabel('Size')

    if output_file != '':
        plt.savefig(os.environ['TEXPATH'] + output_file + '.svg')

    if show_plot:
        plt.show()


def plot_runtime_comparison_sbs(data_cluster, sbs_solver, output_file='', show_plot=False, dpi=192):
    runtimes_sbs = []
    runtimes_cluster_solver = []
    solvers = [cluster['cluster_performance_solver'] for cluster in data_cluster]
    for cluster in data_cluster:
        cluster_solver = cluster['cluster_performance_solver']

        for solver_tuple in cluster['cluster_par2'][0]:
            if solver_tuple[0] == sbs_solver:
                runtimes_sbs.append(solver_tuple[1])
            if solver_tuple[0] == cluster_solver:
                runtimes_cluster_solver.append(solver_tuple[1])

    X_axis = range(len(data_cluster))

    plt.figure(figsize=(1200 / dpi, 600 / dpi), dpi=dpi)

    plt.scatter(X_axis, runtimes_sbs, zorder=2, label='SBS')
    plt.bar(X_axis, runtimes_cluster_solver, zorder=1, label='BPF')

    plt.xlabel('Cluster')
    plt.ylabel('Par2-Score (s)')

    for x, solver in zip(X_axis, solvers):
        plt.annotate(solver, (x, 1000), rotation=90)

    plt.legend()
    plt.tight_layout()

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
        plt.annotate(solver, (x, 1000), rotation=90)

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
        plt.annotate(solver, (x+1, 1000), rotation=90)

    # plt.legend()
    plt.tight_layout()

    if output_file != '':
        plt.savefig(os.environ['TEXPATH'] + output_file + '.svg')

    if show_plot:
        plt.show()

