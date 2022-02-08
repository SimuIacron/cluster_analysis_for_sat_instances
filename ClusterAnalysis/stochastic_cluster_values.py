import numpy as np
from collections import Counter

from numpy import mean

from DataFormats.DbInstance import DbInstance
from run_experiments import write_json, read_json
from util_scripts import util, DatabaseReader
from util_scripts.pareto_optimal import get_pareto_indices

# calculates the mean, variance and std for base, gate and runtimes
# data_clustering
# data_clusters
# db_instance
from util_scripts.scores import par2
from write_to_csv import write_to_csv


def calculate_feature_stochastic(data_clustering, data_clusters, data_dataset, db_instance: DbInstance):
    data_clusters_stochastic = []

    for cluster in data_clusters:
        clustering = get_clustering_for_cluster(data_clustering, cluster)

        base = []
        gate = []
        runtimes = []

        for i, inst in enumerate(clustering['clustering']):
            if inst == cluster['cluster_idx']:
                base.append(db_instance.base_wh[i])
                gate.append(db_instance.gate_wh[i])
                runtimes.append(db_instance.solver_wh[i])

        base_rot = util.rotateNestedLists(base)
        base_variance = [np.var(feature) for feature in base_rot]
        base_mean = [np.mean(feature) for feature in base_rot]
        base_std = [np.std(feature) for feature in base_rot]
        base_min = [np.min(feature) for feature in base_rot]
        base_max = [np.max(feature) for feature in base_rot]
        base_01 = util.rotateNestedLists([scale_array_to_01_to_given_interval(feature, min_, max_)
                                          for feature, min_, max_ in
                                          zip(base_rot, data_dataset['base_min'], data_dataset['base_max'])])

        gate_rot = util.rotateNestedLists(gate)
        gate_variance = [np.var(feature) for feature in gate_rot]
        gate_mean = [np.mean(feature) for feature in gate_rot]
        gate_std = [np.std(feature) for feature in gate_rot]
        gate_min = [np.min(feature) for feature in gate_rot]
        gate_max = [np.max(feature) for feature in gate_rot]
        gate_01 = util.rotateNestedLists([scale_array_to_01_to_given_interval(feature, min_, max_)
                                          for feature, min_, max_ in
                                          zip(gate_rot, data_dataset['gate_min'], data_dataset['gate_max'])])

        runtimes_rot = util.rotateNestedLists(runtimes)
        runtimes_variance = [np.var(feature) for feature in runtimes_rot]
        runtimes_mean = [np.mean(feature) for feature in runtimes_rot]
        runtimes_std = [np.std(feature) for feature in runtimes_rot]
        runtimes_min = [np.min(feature) for feature in runtimes_rot]
        runtimes_max = [np.max(feature) for feature in runtimes_rot]
        runtimes_01 = util.rotateNestedLists([scale_array_to_01_to_given_interval(feature, min_, max_)
                                              for feature, min_, max_ in
                                              zip(runtimes_rot, data_dataset['runtimes_min'],
                                                  data_dataset['runtimes_max'])])

        new_dict = dict(cluster, **{
            'base': base,
            'base_01': base_01,
            'base_variance': base_variance,
            'base_mean': base_mean,
            'base_std': base_std,
            'base_min': base_min,
            'base_max': base_max,
            'gate': gate,
            'gate_01': gate_01,
            'gate_variance': gate_variance,
            'gate_mean': gate_mean,
            'gate_std': gate_std,
            'gate_min': gate_min,
            'gate_max': gate_max,
            'runtimes': runtimes,
            'runtimes_01': runtimes_01,
            'runtimes_variance': runtimes_variance,
            'runtimes_mean': runtimes_mean,
            'runtimes_std': runtimes_std,
            'runtimes_min': runtimes_min,
            'runtimes_max': runtimes_max
        })

        data_clusters_stochastic.append(new_dict)

    return data_clusters_stochastic


# requires calculate_biggest_family_for_cluster to be calculated for the data_clusters
# calculates for each family the best cluster by minimizing/maximizing the given parameter over all clusters
# where the family has a majority of instances in it
# data_clusters: The cluster data
# filter_param: The parameter that should be minimized/maximized
# minimize: Whether the filter_param should be maximized or minimized
def filter_best_cluster_for_each_family(data_clusters, filter_param, minimize=False):
    family_dict = {}
    for cluster in data_clusters:
        cluster_family = cluster['family_list'][0][0]
        if cluster_family not in family_dict:
            family_dict[cluster_family] = cluster
        else:
            if minimize:
                if family_dict[cluster_family][filter_param] > cluster[filter_param]:
                    family_dict[cluster_family] = cluster
            else:
                if family_dict[cluster_family][filter_param] < cluster[filter_param]:
                    family_dict[cluster_family] = cluster

    return family_dict


def get_unsolvable_instances_amount(data_clustering, data_clusters, db_instance:DbInstance):

    unsolvable_clusters = []
    for cluster in data_clusters:
        clustering = get_clustering_for_cluster(data_clustering, cluster)
        unsolvable = 0
        for instance, runtimes in zip(clustering['clustering'], db_instance.solver_wh):
            if instance == cluster['cluster_idx']:
                is_unsolvable = True
                for time in runtimes:
                    if time != DatabaseReader.TIMEOUT:
                        is_unsolvable = False
                        break
                if is_unsolvable:
                    unsolvable = unsolvable + 1
        new_dict = dict(cluster, **{
            'unsolvable instances': unsolvable
        })

        unsolvable_clusters.append(new_dict)

    return unsolvable_clusters



# filters the clusters that are pareto optimal for the given parameters
# data_cluster
# filter_params: List of the parameters that should be used when calculating pareto optimal clusters
# minimize_params: List of booleans, whether the filter_param should be minimized or maximized
def filter_pareto_optimal_clusters(data_clusters, filter_params, minimize_params):
    pareto_data = [[] for _ in filter_params]
    for cluster in data_clusters:
        for i, param in enumerate(filter_params):
            pareto_data[i].append(cluster[param])

    pareto_optimal_clusters = []
    indices = get_pareto_indices(pareto_data, minimize_params)
    for i in indices:
        pareto_optimal_clusters.append(data_clusters[i])

    print('remaining clusters after filtering for pareto optimal clusters with params {b}: {a}'
          .format(a=len(pareto_optimal_clusters), b=filter_params))
    return pareto_optimal_clusters


# calculates the pareto optimal solvers for each cluster in data_clusters_stochastic by minimizing the std and
# maximizing the mean
# data_clusters_stochastic: needs to contain mean and std for the solvers
# db_instance
def calculate_pareto_optimal_solvers_std_mean(data_clusters_stochastic, db_instance: DbInstance):
    data_clusters_pareto_optimal = []
    for cluster in data_clusters_stochastic:
        mean = cluster['runtimes_mean']
        std = cluster['runtimes_std']
        indices = get_pareto_indices([mean, std], [True, True])

        pareto_optimal_solvers = []
        for i in indices:
            pareto_optimal_solvers.append((db_instance.solver_f[i], mean[i], std[i]))

        new_dict = dict(cluster, **{
            'pareto_optimal_solvers': pareto_optimal_solvers
        })

        data_clusters_pareto_optimal.append(new_dict)

    return data_clusters_pareto_optimal


# calculates the cluster_deviation score for each cluster in data_clusters_stochastic
# data_clusters_stochastic: Must contain mean and std
# db_instance
def calculate_cluster_performance_score(data_clusters_stochastic, db_instance: DbInstance):
    data_clusters_stochastic_performance_score = []
    for cluster in data_clusters_stochastic:

        par2_list = [mean([runtime_timeout_par2(runtime, DatabaseReader.TIMEOUT) for runtime in runtimes])
                     for runtimes in util.rotateNestedLists(cluster['runtimes'])]

        cluster_performances = [par2_ + standard_deviation
                                for par2_, standard_deviation in
                                zip(par2_list,
                                    cluster['runtimes_std'])]
        cluster_performance_score = min(cluster_performances)
        bps_index = np.argmin(cluster_performances)
        solver = db_instance.solver_f[bps_index]

        new_dict = dict(cluster, **{
            'cluster_performance_score': cluster_performance_score,
            'cluster_performance_solver': solver,
            'cluster_performances': cluster_performances,
            'cluster_performance_solver_par2': calculate_par2([item[bps_index] for item in cluster['runtimes']],
                                                              DatabaseReader.TIMEOUT)
        })
        data_clusters_stochastic_performance_score.append(new_dict)

    sorted_data = sorted(data_clusters_stochastic_performance_score, key=lambda d: d['cluster_performance_score'])
    return sorted_data


def calculate_factor_of_sbs_and_deviation_solver(data_clustering, data_clusters, sbs_solver, db_instance: DbInstance):
    data_clusters_factor = []
    for cluster in data_clusters:
        sbs_index = db_instance.solver_f.index(sbs_solver)
        performance_index = db_instance.solver_f.index(cluster['cluster_performance_solver'])

        runtimes = util.rotateNestedLists(get_cluster_runtimes(data_clustering, cluster, db_instance))
        par2_sbs = calculate_par2(runtimes[sbs_index], 5000)
        par2_performance = calculate_par2(runtimes[performance_index], 5000)
        factor = par2_sbs / par2_performance

        new_dict = dict(cluster, **{
            'sbs_performance_factor': factor
        })
        data_clusters_factor.append(new_dict)

    return data_clusters_factor


# searches for clusters where all instances in the cluster are unsolvable for all solvers
# data_cluster_stochastic: must contain mean and variance
def search_clusters_with_unsolvable_instances(data_clusters_stochastic):
    unsolvable_clusters = []
    for cluster in data_clusters_stochastic:
        is_unsolvable = True
        for mean_value, variance_value in zip(cluster['runtimes_mean'], cluster['runtimes_variance']):
            if mean_value != DatabaseReader.TIMEOUT:  # or variance_value != 0:
                is_unsolvable = False
                break

        if is_unsolvable:
            unsolvable_clusters.append(cluster)
    return unsolvable_clusters


# filters the clusters by a maximum amount of clusterings
# def filter_clusterings_by_cluster_amount(data_clustering, data_cluster, max_size):
#     data_clusters_filtered = []
#     for clustering in data_clustering:
#         if len(clustering['clusters']) < max_size:
#             id_ = clustering['id']
#             for cluster in data_cluster:
#                 if cluster['id'] == id_:
#                     data_clusters_filtered.append(cluster)
#
#     print('remaining clusters after filtering clustering size: ' + str(len(data_clusters_filtered)))
#     return data_clusters_filtered


# filters the given clusters by parameters as well as the size of the cluster and the amount of clusters in the
# clustering
# data_clustering: List of all clusterings
# list of all clusters to filter
# param_names: the name of all parameters used for filtering, parameters not listed will not be used for filtering, so
# all possible values of them will occur
# param_values_list: The values of the parameters, that are allowed in the filtering, other values will be excluded
# from the filtering
# min_cluster_size: The minimal size of a cluster
# max_cluster_amount: The maximum amount of clusters in the clustering the cluster is part of
def filter_cluster_data(data_clustering, data_cluster, param_names, param_values_list, cluster_size_interval,
                        cluster_amount_interval):
    filtered_data_cluster = []
    for cluster in data_cluster:
        clustering = get_clustering_for_cluster(data_clustering, cluster)

        if cluster_size_interval[0] <= cluster['cluster_size'] <= cluster_size_interval[1] and \
                cluster_amount_interval[0] <= len(clustering['clusters']) <= cluster_amount_interval[1]:

            params_fit = True
            for param_name, param_values in zip(param_names, param_values_list):
                if clustering['settings'][param_name] not in param_values:
                    params_fit = False
                    break

            if params_fit:
                filtered_data_cluster.append(cluster)

    print('remaining clusters after filtering general: ' + str(len(filtered_data_cluster)))
    return filtered_data_cluster


def filter_specific_clustering(data_clusters, id_):
    data_specific_clustering = []
    for cluster in data_clusters:
        if cluster['id'] == id_:
            data_specific_clustering.append(cluster)

    return data_specific_clustering


def calculate_clusters_in_strip(data_clustering, data_clusters, db_instance: DbInstance):
    offset = 3
    grade = 1 / 10

    data_cluster_par2_strip = []

    for cluster in data_clusters:
        clustering = get_clustering_for_cluster(data_clustering, cluster)
        csbs = cluster['cluster_par2'][0][0]
        csbs_strip = (csbs[1] + offset) * (1 + grade)
        cluster_index = cluster['cluster_idx']
        cluster_positions = []
        for i, current_cluster_index in enumerate(clustering['clustering']):
            if current_cluster_index == cluster_index:
                cluster_positions.append(i)

        par2_strip = []
        for solver in db_instance.solver_f:
            par2_ = par2(solver, db_instance, cluster_positions, DatabaseReader.TIMEOUT)
            if par2_ <= csbs_strip:
                par2_strip.append((solver, par2_))

        new_dict = dict(cluster, **{
            'par2_strip': sorted(par2_strip, key=lambda d: d[1])
        })
        data_cluster_par2_strip.append(new_dict)

    return data_cluster_par2_strip


# calculates what family has the majority in a cluster
# data_clustering: List of all clusterings
# data_cluster
# db_instance
def calculate_biggest_family_for_cluster(data_clustering, data_clusters, db_instance: DbInstance):
    data_cluster_family = []

    family_list_total = [item[0] for item in db_instance.family_wh]
    family_count_total = Counter(family_list_total)

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
            'family_biggest_size': family_list[0][1],
            'family_highest_percentage': family_highest_percentage,
            'family_total_percentage': family_list[0][1] / family_count_total[family_list[0][0]]
        })
        data_cluster_family.append(new_dict)

    return data_cluster_family


# selects the base and gate features that have a lower standard deviation in comparison to the factor of
# the standard deviation of all instances.
# data_cluster: The clusters, must contain stochastic values
# dataset_stochastic_values: The stochastic values of all features over the whole dataset
# db_instance
# max_std: The feature gets selected if cluster_std < max_std * dataset_std of a feature
def find_base_and_gate_features_with_low_std(data_cluster, dataset_stochastic_values, db_instance: DbInstance,
                                             max_std=0.1):
    data_cluster_interesting_features = []
    for cluster in data_cluster:
        interesting_features_base = []
        for i, (cluster_std, dataset_std) in enumerate(zip(cluster['base_std'], dataset_stochastic_values['base_std'])):
            if cluster_std < max_std * dataset_std:
                feature = db_instance.base_f[i]
                interesting_features_base.append(
                    (feature, cluster['base_mean'][i], cluster_std, cluster_std / dataset_std))
        interesting_features_gate = []
        for i, (cluster_std, dataset_std) in enumerate(zip(cluster['gate_std'], dataset_stochastic_values['gate_std'])):
            if cluster_std < max_std * dataset_std and dataset_std != 0:
                feature = db_instance.gate_f[i]
                interesting_features_gate.append(
                    (feature, cluster['gate_mean'][i], cluster_std, cluster_std / dataset_std))

        new_dict = dict(cluster, **{
            'low_std_base': interesting_features_base,
            'low_std_base': interesting_features_base,
            'low_std_gate': interesting_features_gate
        })
        data_cluster_interesting_features.append(new_dict)

    return data_cluster_interesting_features


# sorts the given clusters after the given parameters
# data_cluster: Must contain the given sort_param
# sort_param: The parameter that is used to sort
# descending: Whether to sort ascending or descending
def sort_after_param(data_cluster, sort_param, descending=False):
    return sorted(data_cluster, key=lambda x: x[sort_param], reverse=descending)


# uses the best solver with the lowest deviation score on all instances in the whole dataset,
# that has a majority in the cluster
# data_cluster: Must contain cluster_performance_solver and family_list
# db_instance
def check_performance_for_all_instances_of_major_family(data_clustering, data_cluster, db_instance: DbInstance):
    database_cluster_complete_family = []
    for cluster in data_cluster:
        clustering = get_clustering_for_cluster(data_clustering, cluster)
        solver = cluster['cluster_performance_solver']
        solver_index = db_instance.solver_f.index(solver)
        family = cluster['family_list'][0][0]
        unsolvable_instances = 0
        runtimes = []
        runtimes_of_family_in_cluster = []
        for i, (current_family, current_runtimes) in enumerate(zip(db_instance.family_wh, db_instance.solver_wh)):
            if current_family[0] == family:
                current_runtime = current_runtimes[solver_index]
                runtimes.append(current_runtime)
                if current_runtime >= DatabaseReader.TIMEOUT:
                    unsolvable_instances = unsolvable_instances + 1
                if clustering['clustering'][i] == cluster['cluster_idx']:
                    runtimes_of_family_in_cluster.append(current_runtime)

        par2 = calculate_par2(runtimes, DatabaseReader.TIMEOUT)
        family_in_cluster_par2 = calculate_par2(runtimes_of_family_in_cluster, DatabaseReader.TIMEOUT)
        new_dict = dict(cluster, **{
            'complete_family_par2': par2,
            'complete_family_unsolvable_instances': unsolvable_instances,
            'complete_family_unsolvable_instances_percentage':
                unsolvable_instances / len(runtimes),
            'family_in_cluster_par2': family_in_cluster_par2
        })
        database_cluster_complete_family.append(new_dict)

    return database_cluster_complete_family


def check_performance_for_instances_with_similar_feature_values(data_cluster, data_clustering, db_instance: DbInstance,
                                                                only_allow_features_that_were_used_in_clustering=True):
    database_cluster_similar_instances = []
    for cluster in data_cluster:
        clustering = get_clustering_for_cluster(data_clustering, cluster)
        selected_data = clustering['settings']['selected_data']

        solver = cluster['cluster_performance_solver']
        solver_index = db_instance.solver_f.index(solver)

        unsolvable_instances = 0
        new_instances_not_in_cluster = 0

        contains_base = True
        contains_gate = True
        if only_allow_features_that_were_used_in_clustering:
            contains_base = all(item in selected_data for item in db_instance.base_f)
            contains_gate = all(item in selected_data for item in db_instance.gate_f)

        low_std_base_feature_index = [db_instance.base_f.index(item[0]) for item in cluster['low_std_base']]
        low_std_gate_feature_index = [db_instance.gate_f.index(item[0]) for item in cluster['low_std_gate']]

        new_instance_families = []
        runtimes_of_instances_close_to_cluster = []
        for i, (base, gate, runtimes, family) in \
                enumerate(zip(db_instance.base_wh, db_instance.gate_wh, db_instance.solver_wh, db_instance.family_wh)):
            near_cluster = True
            if contains_base:
                for index in low_std_base_feature_index:
                    if not (cluster['base_max'][index] >= base[index] >= cluster['base_min'][index]):
                        near_cluster = False
                        break
            if contains_gate and near_cluster:
                for index in low_std_gate_feature_index:
                    if not (cluster['gate_max'][index] >= gate[index] >= cluster['gate_min'][index]):
                        near_cluster = False
                        break

            if near_cluster:
                if clustering['clustering'][i] != cluster['cluster_idx']:
                    new_instances_not_in_cluster = new_instances_not_in_cluster + 1
                    new_instance_families.append(family[0])

                current_runtime = runtimes[solver_index]
                runtimes_of_instances_close_to_cluster.append(current_runtime)
                if current_runtime >= DatabaseReader.TIMEOUT:
                    unsolvable_instances = unsolvable_instances + 1

        par2 = calculate_par2(runtimes_of_instances_close_to_cluster, DatabaseReader.TIMEOUT)
        new_dict = dict(cluster, **{
            'similar_instances_par2': par2,
            'similar_instances_unsolvable_instances': unsolvable_instances,
            'similar_instances_unsolvable_instances_percentage':
                unsolvable_instances / len(runtimes_of_instances_close_to_cluster),
            'similar_instances_new_in_cluster': new_instances_not_in_cluster,
            'similar_instances_new_in_cluster_families': Counter(new_instance_families)
        })
        database_cluster_similar_instances.append(new_dict)

    return database_cluster_similar_instances


def filter_non_clusters(data_cluster):
    data_cluster_filtered = []
    for cluster in data_cluster:
        if cluster['cluster_idx'] != -1:
            data_cluster_filtered.append(cluster)

    print('remaining clusters after filtering non clusters: {a}'.format(a=len(data_cluster_filtered)))
    return data_cluster_filtered


def filter_same_cluster(data_clustering, data_cluster):
    remaining_clusters = []
    accepted_clusters = []
    for cluster in data_cluster:
        clustering = get_clustering_for_cluster(data_clustering, cluster)
        index_list = []
        for i, value in enumerate(clustering['clustering']):
            if value == cluster['cluster_idx']:
                index_list.append(i)
        already_in_list = False
        for other_cluster in accepted_clusters:
            if Counter(other_cluster) == Counter(index_list):
                # print('{a} same as {b}'.format(a=other_cluster, b=index_list))
                already_in_list = True
                break

        if not already_in_list:
            accepted_clusters.append(index_list)
            remaining_clusters.append(cluster)

    print('remaining clusters after filtering equals: {a}'.format(a=len(remaining_clusters)))
    return remaining_clusters


def find_best_clustering_by_performance_score(data_clustering, data_clusters):
    clustering_list = []
    for clustering in data_clustering:
        id_ = clustering['id']
        count = 0
        size = 0
        clustering_performance_score = 0
        for cluster in data_clusters:
            if cluster['id'] == id_:
                count = count + 1
                size = size + cluster['cluster_size']
                clustering_performance_score = clustering_performance_score + cluster['cluster_performance_score'] * \
                                               cluster['cluster_size']

        if count != 0:
            clustering_performance_score = clustering_performance_score / size
            assert count == len(
                clustering['clusters']), '{id}: added {a} clusters to the score, but expected {b}'.format(
                id=id_, a=count, b=len(clustering['clusters']))
            new_dict = dict({
                'clustering_performance_score': clustering_performance_score,
                'size': len(clustering['clusters']),
                'cluster_sizes': Counter(clustering['clustering'])
            }, **clustering)
            clustering_list.append(new_dict)

    return clustering_list


def sort_clusters_by_lowest_performance_scores_of_best_clusters(data_clustering, data_clusters, min_cluster_size=20,
                                                                sbs_solver='', use_best_n=3):
    clusters_mapped_to_clustering = {}
    for cluster in data_clusters:
        if cluster['id'] not in clusters_mapped_to_clustering:
            clusters_mapped_to_clustering[cluster['id']] = []
        clusters_mapped_to_clustering[cluster['id']].append(cluster)

    clusterings = []
    for key, item in clusters_mapped_to_clustering.items():
        clustering = get_clustering_for_cluster(data_clustering, item[0])
        assert len(clustering['clusters']) == len(item), 'expected {a}, but found {b} clusters'.format(
            a=len(clustering['clusters']), b=len(item))

        if sbs_solver != '':
            removed_sbs = []
            for cluster in item:
                if cluster['cluster_performance_solver'] != sbs_solver:
                    removed_sbs.append(cluster)
            item = removed_sbs

        filtered_item = []
        for cluster in item:
            if cluster['cluster_size'] >= min_cluster_size:
                filtered_item.append(cluster)

        best_clusters = sorted(filtered_item, key=lambda d: d['cluster_performance_score'])[:use_best_n]
        clustering_score = 0
        size = 0
        for cluster in best_clusters:
            clustering_score = clustering_score + cluster['cluster_performance_score'] * cluster['cluster_size']
            size = size + cluster['cluster_size']

        if size != 0:
            clustering_score = clustering_score / size

            new_dict = dict({
                'clustering_performance_score_n_best': clustering_score,
                'n_best': len(best_clusters),
                'size': len(clustering['clusters']),
                'cluster_sizes': Counter(clustering['clustering'])
            }, **clustering)
            clusterings.append(new_dict)

    return sorted(clusterings, key=lambda d: d['clustering_performance_score_n_best'])


def filter_clusters_where_sbs_and_bps_are_different(data_cluster):
    filtered = []
    for cluster in data_cluster:
        if cluster['cluster_performance_solver'] != cluster['cluster_par2'][0][0][0]:
            filtered.append(cluster)

    return filtered


# --- Helper functions -------------------------------------------------------------------------------------------------

# calculates the par2 score for the given runtimes using the timeout value given
def calculate_par2(runtimes, timeout):
    par2 = 0
    for runtime in runtimes:
        par2 = par2 + runtime_timeout_par2(runtime, timeout)
    return par2 / len(runtimes)


def runtime_timeout_par2(runtime, timeout):
    if runtime >= timeout:
        return timeout * 2
    else:
        return runtime


# gets the runtimes for the instances in the cluster
# data_clustering: list of clustering dicts
# cluster: cluster dict
# db_instance
def get_cluster_runtimes(data_clustering, cluster, db_instance: DbInstance):
    clustering = get_clustering_for_cluster(data_clustering, cluster)

    cluster_runtimes = []
    for i, inst in enumerate(clustering['clustering']):
        if inst == cluster['cluster_idx']:
            cluster_runtimes.append(db_instance.solver_wh[i])

    return cluster_runtimes


# gets the clustering to the given cluster
# data_clustering: List of all clusterings
# cluster: The cluster we want the clustering for
def get_clustering_for_cluster(data_clustering, cluster):
    idx = cluster['id']
    clustering = data_clustering[idx]
    assert clustering['id'] == idx, 'clustering had id "{a}" but expected id "{b}"'.format(a=clustering['id'],
                                                                                           b=idx)
    return clustering


# exports a file with the clusters and their mean/variance/standard_deviation
# input_file_clustering: file in which the clustering dicts are saved
# input_file_cluster: file in which the clusters are saved
# output_file: the location where the finished data is saved
# db_instance
def export_variance_mean_of_cluster(input_file_clustering, input_file_cluster, output_file, db_instance: DbInstance):
    data_clustering = read_json(input_file_clustering)
    data_cluster = read_json(input_file_cluster)
    variance_mean_list = calculate_feature_stochastic(data_clustering, data_cluster, db_instance)
    write_json(output_file, variance_mean_list)


# scales given array values to a scale of 0 to 1
# example: [3,5,7] --> [0, 0.5, 1]
def scale_array_to_01_to_given_interval(array, min_v, max_v):
    if max_v - min_v != 0:
        scaled_array = [((value - min_v) / (max_v - min_v)) for value in array]
    else:
        scaled_array = [0] * len(array)

    return scaled_array


def generate_csv_cluster_strip(data_clusters, header, file):
    export_list = []
    for i, cluster in enumerate(data_clusters):
        strip = cluster['par2_strip']
        text = ''
        for elem in strip:
            text = text + elem[0] + ', '

        text = text.replace("_", "-")
        export_list.append([i, text[:-2]])

    write_to_csv(file, header, export_list)
