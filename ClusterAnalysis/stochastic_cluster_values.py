import numpy as np
from collections import Counter
from DataFormats.DbInstance import DbInstance
from run_experiments import write_json, read_json
from util_scripts import util, DatabaseReader
from util_scripts.pareto_optimal import get_pareto_indices_2d, get_pareto_indices


def calculate_feature_stochastic(data_clustering, data_clusters, db_instance: DbInstance):
    data_clusters_stochastic = []

    base_interval_size = [max(item) for item in util.rotateNestedLists(db_instance.base_wh)]
    gate_interval_size = [max(item) for item in util.rotateNestedLists(db_instance.gate_wh)]
    # for i in range(len(base_interval_size)):
    #     if base_interval_size[i] == 0:
    #         base_interval_size[i] = float(1)
    #
    # for i in range(len(gate_interval_size)):
    #     if gate_interval_size[i] == 0:
    #         gate_interval_size[i] = float(1)

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

        gate_rot = util.rotateNestedLists(gate)
        gate_variance = [np.var(feature) for feature in gate_rot]
        gate_mean = [np.mean(feature)for feature in gate_rot]
        gate_std = [np.std(feature)for feature in gate_rot]

        runtimes_rot = util.rotateNestedLists(runtimes)
        runtimes_variance = [np.var(feature) for feature in runtimes_rot]
        runtimes_mean = [np.mean(feature) for feature in runtimes_rot]
        runtimes_std = [np.std(feature) for feature in runtimes_rot]

        new_dict = dict(cluster, **{
            'base_variance': base_variance,
            'base_mean': base_mean,
            'base_std': base_std,
            'base_interval_size': base_interval_size,
            'gate_variance': gate_variance,
            'gate_mean': gate_mean,
            'gate_std': gate_std,
            'gate_interval_size': gate_interval_size,
            'runtimes_variance': runtimes_variance,
            'runtimes_mean': runtimes_mean,
            'runtimes_std': runtimes_std
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


# filters the clusters that are pareto optimal for the given parameters
# data_cluster
# filter_params: List of the parameters that should be used when calculating pareto optimal clusters
# minimize_params: List of booleans, whether the filter_param should be minimized or maximized
def filter_pareto_optimal_clusters(data_clusters, filter_params, minimize_params):
    pareto_data = [[] for i in filter_params]
    for cluster in data_clusters:
        for i, param in enumerate(filter_params):
            pareto_data[i].append(cluster[param])

    pareto_optimal_clusters = []
    indices = get_pareto_indices(pareto_data, minimize_params)
    for i in indices:
        pareto_optimal_clusters.append(data_clusters[i])

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
        indices = get_pareto_indices_2d(mean, std, minimize_x=True,
                                        minimize_y=True)

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
def calculate_cluster_deviation_score(data_clusters_stochastic, db_instance: DbInstance):
    data_clusters_stochastic_deviation_score = []
    for cluster in data_clusters_stochastic:
        cluster_deviations = [mean_value + standard_deviation for mean_value, standard_deviation in
                              zip(cluster['runtimes_mean'],
                                  cluster['runtimes_std'])]
        cluster_deviation_score = min(cluster_deviations)
        solver = db_instance.solver_f[np.argmin(cluster_deviations)]
        new_dict = dict(cluster, **{
            'cluster_deviation_score': cluster_deviation_score,
            'cluster_deviation_solver': solver,
            'cluster_deviations': cluster_deviations
        })
        data_clusters_stochastic_deviation_score.append(new_dict)

    sorted_data = sorted(data_clusters_stochastic_deviation_score, key=lambda d: d['cluster_deviation_score'])
    return sorted_data


# searches for clusters where all instances in the cluster are unsolvable for all solvers
# data_cluster_stochastic: must contain mean and variance
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


# --- Helper functions -------------------------------------------------------------------------------------------------

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
