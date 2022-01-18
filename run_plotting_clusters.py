import os
from collections import Counter
from math import ceil

import numpy as np
from matplotlib import pyplot as plt
from numpy import mean

import util_scripts.util
from DataAnalysis.Evaluation.scoring_util import convert_families_to_int
from DataFormats.DbInstance import DbInstance
from run_experiments import read_json, write_json
from util_scripts import DatabaseReader


def export_clusters_sorted_best(input_file_par2, output_file):
    data = read_json(input_file_par2)
    exportList = []
    for evaluation in data:
        cluster_sizes = Counter(evaluation['clustering'])
        for cluster, size in cluster_sizes.items():
            cluster_dict = {'id': evaluation['id'],
                            'cluster_idx': cluster,
                            'cluster_par2': evaluation['par2'][1][str(cluster)],
                            'cluster_size': size}
            exportList.append(cluster_dict)

    sorted_export_list = sorted(exportList, key=lambda d: d['cluster_par2'])
    write_json(output_file, sorted_export_list)


def plot_biggest_cluster_for_family(input_file_clustering, input_file_cluster,
                                    param_names, param_values_list, min_cluster_size, max_cluster_amount,
                                    db_instance: DbInstance, min_percentage=0.9, output_file='', show_plot=False,
                                    dpi=192, comparison='SIZE'):
    data_clustering = read_json(input_file_clustering)
    data_cluster = read_json(input_file_cluster)
    filtered_data = filter_cluster_data(data_clustering, data_cluster,
                                        param_names, param_values_list, min_cluster_size, max_cluster_amount)

    family_dict = {}

    for cluster in filtered_data:
        idx = cluster['id']
        clustering = data_clustering[idx]
        assert clustering['id'] == idx

        families = []
        for idx, elem in enumerate(clustering['clustering']):
            if elem == cluster['cluster_idx']:
                families.append(db_instance.family_wh[idx][0])

        cluster_family_count = Counter(families)
        total_family_count = Counter([family[0] for family in db_instance.family_wh])
        for key, item in cluster_family_count.items():
            percentage = item / cluster['cluster_size']
            if percentage > min_percentage:
                new_dict = dict(cluster, **{
                    'family': key,
                    'percentage_of_family_in_cluster': percentage,
                    'occurrence_of_family_in_cluster': item,
                    'percentage_of_total_family_instances_in_cluster': item / total_family_count[key]
                })
                if key not in family_dict:
                    family_dict[key] = new_dict
                else:
                    if comparison == 'SIZE':
                        if family_dict[key]['occurrence_of_family_in_cluster'] < new_dict[
                            'occurrence_of_family_in_cluster']:
                            family_dict[key] = new_dict
                    elif comparison == 'RUNTIME':
                        if family_dict[key]['cluster_par2'][0][0][1] > new_dict['cluster_par2'][0][0][1]:
                            family_dict[key] = new_dict
                    else:
                        assert False

    lowest_runtime = []
    lowest_runtime_family_instances = []
    percentage_of_family_in_cluster = []
    percentage_of_total_family_instances_in_cluster = []
    families_in_plot = []
    for key, item in family_dict.items():
        families_in_plot.append(key)

        # calculates par2 score only for the instances of the class in the cluster
        idx = item['id']
        clustering = data_clustering[idx]
        assert clustering['id'] == idx
        par2_scores = [[]] * len(db_instance.solver_f)
        for idx, value in enumerate(clustering['clustering']):
            if db_instance.family_wh[idx][0] == item['family'] and value == item['cluster_idx']:
                for idx2, time in enumerate(db_instance.solver_wh[idx]):
                    if time == DatabaseReader.TIMEOUT:
                        par2_scores[idx2].append(time * 2)
                    else:
                        par2_scores[idx2].append(time)
        min_par2 = 10000
        for scores in par2_scores:
            if mean(scores) < min_par2:
                min_par2 = mean(scores)

        lowest_runtime_family_instances.append(min_par2)
        lowest_runtime.append(item['cluster_par2'][0][0][1])
        percentage_of_family_in_cluster.append(item['percentage_of_family_in_cluster'])
        percentage_of_total_family_instances_in_cluster.append(item['percentage_of_total_family_instances_in_cluster'])

    X_axis = np.arange(len(families_in_plot))

    fig, ax1 = plt.subplots(figsize=(1700 / dpi, 1000 / dpi), dpi=dpi)
    ax1.set_xlabel('Families')
    plt.xticks(X_axis, families_in_plot, rotation=90)
    ax1.set_ylabel('Runtimes in Par2 (s)')
    ax1.bar(X_axis - 0.1, lowest_runtime, 0.2, label='Cluster Runtime', color='tab:red')
    ax1.bar(X_axis - 0.3, lowest_runtime_family_instances, 0.2, label='Family Runtime', color='tab:green')
    ax1.set_ylim((0, 11000))

    ax2 = ax1.twinx()
    ax2.set_ylabel('Percentage (%)')
    ax2.bar(X_axis + 0.1, percentage_of_family_in_cluster, 0.2, label='Percentage in Cluster')
    ax2.bar(X_axis + 0.3, percentage_of_total_family_instances_in_cluster, 0.2, label='Percentage in Total')
    ax2.legend(loc=1)
    ax1.legend(loc=2)

    fig.tight_layout()

    if output_file != '':
        plt.savefig(os.environ['TEXPATH'] + output_file + '.svg')

    if show_plot:
        plt.show()


def calculate_biggest_family_for_cluster(data_clustering, data_clusters, db_instance: DbInstance):
    data_cluster_family = []
    for cluster in data_clusters:
        family_list = []
        clustering = get_clustering_for_cluster(data_clustering, cluster)
        for i, value in enumerate(clustering['clustering']):
            if value == cluster['cluster_idx']:
                family_list.append(db_instance.family_wh[i][0])

        family_count = Counter(family_list)
        family_list = sorted([(key, value) for key, value in family_count.items()], key=lambda d: d[1], reverse=True)
        family_highest_percentage = family_list[0][1] / cluster['cluster_size']
        new_dict = dict(cluster, **{
            'family_list': family_list,
            'family_highest_percentage': family_highest_percentage
        })
        data_cluster_family.append(new_dict)

    return data_cluster_family


def calculate_cluster_deviation_score(data_clusters_stochastic):
    data_clusters_stochastic_deviation_score = []
    for cluster in data_clusters_stochastic:
        cluster_deviations = [mean_value + standard_deviation for mean_value, standard_deviation in
                              zip(cluster['runtimes_mean'],
                                  cluster['runtimes_standard_deviation'])]
        cluster_deviation_score = min(cluster_deviations)
        new_dict = dict(cluster, **{
            'cluster_deviation_score': cluster_deviation_score,
            'cluster_deviations': cluster_deviations
        })
        data_clusters_stochastic_deviation_score.append(new_dict)

    sorted_data = sorted(data_clusters_stochastic_deviation_score, key=lambda d: d['cluster_deviation_score'])
    return sorted_data


def search_clusters_with_unsolvable_instances(data_clusters_stochastic):
    unsolvable_clusters = []
    for cluster in data_clusters_stochastic:
        is_unsolvable = True
        for mean_value, variance_value in zip(cluster['runtimes_mean'], cluster['runtime_variance']):
            if mean_value != DatabaseReader.TIMEOUT or variance_value != 0:
                is_unsolvable = False
                break

        if is_unsolvable:
            unsolvable_clusters.append(cluster)
    return unsolvable_clusters


def get_clustering_for_cluster(data_clustering, cluster):
    idx = cluster['id']
    clustering = data_clustering[idx]
    assert clustering['id'] == idx, 'clustering had id "{a}" but expected id "{b}"'.format(a=clustering['id'],
                                                                                           b=idx)
    return clustering


def filter_cluster_data(data_clustering, data_cluster, param_names, param_values_list, min_cluster_size,
                        max_cluster_amount):
    print('start filtering')
    filtered_data_cluster = []
    for cluster in data_cluster:
        clustering = get_clustering_for_cluster(data_clustering, cluster)

        if cluster['cluster_size'] >= min_cluster_size and len(clustering['clusters']) <= max_cluster_amount:

            params_fit = True
            for param_name, param_values in zip(param_names, param_values_list):
                if clustering['settings'][param_name] not in param_values:
                    params_fit = False
                    break

            if params_fit:
                filtered_data_cluster.append(cluster)

    print('finished filtering')
    print('remaining clusters: ' + str(len(filtered_data_cluster)))
    return filtered_data_cluster


def compare_with_family(input_file_clustering, input_file_cluster, param_names, param_values_list, min_cluster_size,
                        max_cluster_amount,
                        db_instance: DbInstance, columns=5, dpi=192, output_file='', show_plot=False, y_axis_size=None):
    data_clustering = read_json(input_file_clustering)
    data_cluster = read_json(input_file_cluster)
    filtered_data = filter_cluster_data(data_clustering, data_cluster,
                                        param_names, param_values_list, min_cluster_size, max_cluster_amount)

    family_cluster_list = []
    family_dict = {}

    for cluster in filtered_data:
        idx = cluster['id']
        clustering = data_clustering[idx]
        assert clustering['id'] == idx

        family = []
        for idx, elem in enumerate(clustering['clustering']):
            if elem == cluster['cluster_idx']:
                family.append(db_instance.family_wh[idx][0])

        family_count = Counter(family)
        highest_percentage = 0
        highest_family = ''
        for key, item in family_count.items():
            percentage = item / cluster['cluster_size']

            if percentage > highest_percentage:
                highest_percentage = percentage
                highest_family = key

        new_dict = dict(cluster, **{
            'family': family,
            'percentage_highest_family': highest_percentage,
            'name_highest_family': highest_family
        })

        if highest_family not in family_dict:
            family_dict[highest_family] = []
        family_dict[highest_family].append(highest_percentage)
        family_cluster_list.append(new_dict)

    families = [key for key, item in family_dict.items()]
    rows = ceil(len(families) / columns)
    n_bins = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
    fig, axes = plt.subplots(nrows=rows, ncols=columns, figsize=(1700 / dpi, 1000 / dpi), dpi=dpi,
                             constrained_layout=True)
    axes_flat = [axes]
    if len(families) > 1:
        axes_flat = axes.flat

    for idx, family in enumerate(families):
        axes_flat[idx].set_title(family)
        if y_axis_size is not None:
            axes_flat[idx].set_ylim(y_axis_size)
        axes_flat[idx].set_xlabel('Percentage of occurence in cluster')
        axes_flat[idx].set_ylabel('frequency')
        axes_flat[idx].hist(family_dict[family], n_bins)

    for idx in range(len(families), len(axes_flat)):
        axes_flat[idx].set_visible(False)

    if output_file != '':
        plt.savefig(os.environ['TEXPATH'] + output_file + '.svg')

    if show_plot:
        plt.show()