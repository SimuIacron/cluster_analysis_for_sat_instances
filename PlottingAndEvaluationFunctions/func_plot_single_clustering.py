import os
from collections import Counter

import numpy as np
from matplotlib import pyplot as plt

import UtilScripts.util
from DataFormats.DbInstance import DbInstance


# For a clustering plots the size of each cluster together with the share of the biggest family and some infos for the
# biggest family in each cluster
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

    X_axis = range(len(biggest_family_size))

    plt.figure(figsize=(1200 / dpi, 1000 / dpi), dpi=dpi)
    plt.barh(X_axis, biggest_family_size, label='Biggest Family')
    plt.barh(X_axis, other_families_size, label='Other Families', left=biggest_family_size)

    for x, family in zip(X_axis, biggest_family_name):
        plt.annotate(
            '{family} ({size}/{total_size}), {percentage}% of family'.format(family=family, size=biggest_family_size[x],
                                                                             total_size=biggest_family_size[x] +
                                                                                        other_families_size[
                                                                                            x], percentage=round(
                    percentage_familes[x] * 100)),
            (150, x + 0.1), rotation=0)

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


# For a clustering plots the size of each cluster together with the share of the biggest family and some infos for the
# biggest family in each cluster
def plot_family_distribution_of_clusters_v2(data_clusters, db_instance: DbInstance, output_file='',
                                            show_plot=False, dpi=192, family_min_percentage = 0.2):

    family_sizes = Counter([family[0] for family in db_instance.family_wh])
    other_familes_label = 'other families'

    biggest_families_size_all = []
    biggest_families_name_all = []
    max_biggest_families = 0
    for cluster in data_clusters:
        biggest_families_size = [0]
        biggest_families_name = [other_familes_label]

        for family_name, family_size in cluster['family_list']:
            if family_size / cluster['cluster_size'] >= family_min_percentage:
                biggest_families_size.append(family_size)
                biggest_families_name.append(family_name)
            else:
                biggest_families_size[0] = biggest_families_size[0] + family_size

        if len(biggest_families_size) > max_biggest_families:
            max_biggest_families = len(biggest_families_size)

        biggest_families_size_all.append(biggest_families_size)
        biggest_families_name_all.append(biggest_families_name)

    for sizes, names in zip(biggest_families_size_all, biggest_families_name_all):
        while len(sizes) < max_biggest_families:
            sizes.append(0)
            names.append('')

    plot_sizes = UtilScripts.util.rotateNestedLists(biggest_families_size_all)
    plot_names = UtilScripts.util.rotateNestedLists(biggest_families_name_all)

    plot_sizes = plot_sizes[1:] + [plot_sizes[0]]
    plot_names = plot_names[1:] + [plot_names[0]]

    X_axis = range(len(plot_sizes[0]))

    plt.figure(figsize=(1200 / dpi, 1000 / dpi), dpi=dpi)
    bottom = [0] * len(plot_sizes[0])
    for elem in plot_sizes:
        plt.barh(X_axis, elem, left=bottom)
        bottom = [a + b for a, b in zip(bottom, elem)]

    texts = [''] * len(plot_sizes[0])
    for x, families, sizes in zip(X_axis, plot_names, plot_sizes):
        for i, (family, size, cluster) in enumerate(zip(families, sizes, data_clusters)):
            cluster_size = cluster['cluster_size']
            if size != 0:
                if family == other_familes_label:
                    texts[i] = texts[i] + '({family}, {size}/{cluster_size}, -), '\
                        .format(family=families[0], size=size, cluster_size=cluster_size)
                else:
                    texts[i] = texts[i] + '({family}, {size}/{cluster_size}, {percentage}%), '\
                        .format(family=family, size=size, cluster_size=cluster_size, percentage=round(size/family_sizes[family] * 100))

    plt.annotate('Legend: (family, share in cluster, percentage of all family instances)', (20, -1), rotation=0)

    for x, text in zip(X_axis, texts):
        plt.annotate(text, (50, x + 0.1), rotation=0)

        # plt.annotate(
        #     '{family} ({size}/{total_size}), {percentage}% of family'.format(family=family, size=biggest_family_size[x],
        #                                                                      total_size=biggest_family_size[x] +
        #                                                                                 other_families_size[
        #                                                                                     x], percentage=round(
        #             percentage_familes[x] * 100)),
        #     (150, x + 0.1), rotation=0)

    # plt.legend(loc='upper right')
    plt.ylabel('Cluster')
    plt.xlabel('Size')
    plt.tight_layout()
    plt.yticks(X_axis)
    plt.gca().invert_yaxis()

    if output_file != '':
        plt.savefig(os.environ['TEXPATH'] + output_file + '.svg')

    if show_plot:
        plt.show()


# for a clustering plots the csbs score of each cluster and the speed of the sbs on the cluster.
# annotates the csbs as well as the solvers in the strip of the csbs
def plot_runtime_comparison_sbs(data_cluster, sbs_solver, output_file='', show_plot=False, dpi=192):
    runtimes_sbs = []
    runtimes_cluster_solver = []
    solvers = [cluster['par2_strip'] for cluster in data_cluster]
    for cluster in data_cluster:

        # cluster_solver = cluster['cluster_par2']
        runtimes_cluster_solver.append(cluster['cluster_par2'][0][0][1])

        for solver_tuple in cluster['cluster_par2'][0]:
            if solver_tuple[0] == sbs_solver:
                runtimes_sbs.append(solver_tuple[1])

    X_axis = range(len(runtimes_sbs))

    plt.figure(figsize=(1200 / dpi, 1000 / dpi), dpi=dpi)

    plt.scatter(runtimes_sbs, X_axis, zorder=2, label='SBS')
    plt.barh(X_axis, runtimes_cluster_solver, zorder=1, label='CSBS')

    plt.ylabel('Cluster')
    plt.xlabel('Par2-Score (s)')

    for x, strip in zip(X_axis, solvers):
        plt.annotate(strip[0][0], (1000, x + 0.1), rotation=0, weight='bold')
        strip_text = ''
        for i, current_solver in enumerate(strip[1:]):
            if i < 5:
                strip_text = strip_text + ', ' + current_solver[0]
            else:
                strip_text = strip_text + ', ' + str(len(strip[1:]) - i) + ' more solvers'
                break

        plt.annotate(strip_text, (1000 + 92 * len(strip[0][0]), x + 0.1), rotation=0)

    plt.legend(loc='upper right')
    plt.tight_layout()
    plt.yticks(X_axis)
    plt.gca().invert_yaxis()

    if output_file != '':
        plt.savefig(os.environ['TEXPATH'] + output_file + '.svg')

    if show_plot:
        plt.show()


def plot_cluster_distribution_for_families(data_clusters, db_instance: DbInstance, output_file='',
                                           show_plot=False, dpi=192):
    family_list = [family[0] for family in db_instance.family_wh]
    count = Counter(family_list)
    count_list = [item for key, item in count.items()]
    quantile = np.quantile(count_list, 0.75)

    Y_axis = []
    for key, item in count.items():
        if item > quantile:
            Y_axis.append(key)

    plt.figure(figsize=(1200 / dpi, 1000 / dpi), dpi=dpi)
    plt.ylabel('Families')
    plt.xlabel('Share in cluster')

    data = []
    for current_family in Y_axis:
        family_data = []
        for cluster in data_clusters:
            for (family, size) in cluster['family_list']:
                if family == current_family:
                    family_data.append(size)

        zero_padding = [0] * (len(data_clusters) - len(family_data))
        family_data = UtilScripts.util.scale_array_to_add_to_1(sorted(family_data, reverse=True)) + zero_padding
        data.append(family_data)

    data, Y_axis = zip(*sorted(zip(data, Y_axis), key=lambda d: d[1], reverse=True))
    data, Y_axis = zip(*sorted(zip(data, Y_axis), key=lambda d: d[0][0], reverse=True))

    data_scaled = UtilScripts.util.rotateNestedLists(data)

    bottom = [0] * len(Y_axis)
    for i, cluster_data in enumerate(data_scaled):
        plt.barh(Y_axis, cluster_data, label=i, left=bottom)
        bottom = [a + b for a, b in zip(bottom, cluster_data)]

    plt.tight_layout()
    plt.yticks(Y_axis)
    plt.gca().invert_yaxis()

    if output_file != '':
        plt.savefig(os.environ['TEXPATH'] + output_file + '.svg')

    if show_plot:
        plt.show()
