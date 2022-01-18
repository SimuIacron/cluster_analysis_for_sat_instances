import numpy as np

from DataFormats.DbInstance import DbInstance
from run_experiments import write_json, read_json


# gets the runtimes for the instances in the cluster
# data_clustering: list of clustering dicts
# cluster: cluster dict
# db_instance
from util_scripts import util


def get_cluster_runtimes(data_clustering, cluster, db_instance: DbInstance):
    id = cluster['id']
    clustering = data_clustering[id]
    assert clustering['id'] == id

    cluster_runtimes = []
    for i, inst in enumerate(clustering['clustering']):
        if inst == cluster['cluster_idx']:
            cluster_runtimes.append(db_instance.solver_wh[i])

    return cluster_runtimes


# exports a file with the clusters and their mean/variance/standard_deviation
# input_file_clustering: file in which the clustering dicts are saved
# input_file_cluster: file in which the clusters are saved
# output_file: the location where the finished data is saved
# db_instance
def export_variance_mean_of_cluster(input_file_clustering, input_file_cluster, output_file, db_instance: DbInstance):
    data_clustering = read_json(input_file_clustering)
    data_cluster = read_json(input_file_cluster)
    variance_mean_list = calculate_variance_mean_of_clusters(data_clustering, data_cluster, db_instance)
    write_json(output_file, variance_mean_list)


# calculates the variance/mean/standard_deviation for the given clusters
# data_clustering: list of clustering dicts where the clusters are in
# data_cluster: list of clusters
# db_instance
def calculate_variance_mean_of_clusters(data_clustering, data_cluster, db_instance: DbInstance):
    print('start calculating mean and variance')
    finished_cluster_data = []
    for cluster in data_cluster:
        cluster_runtimes = get_cluster_runtimes(data_clustering, cluster, db_instance)
        runtimes_per_solver = util.rotateNestedLists(cluster_runtimes)
        runtimes_means = [np.mean(item) for item in runtimes_per_solver]
        runtimes_variance = [np.var(item) for item in runtimes_per_solver]
        runtimes_standard_deviation = [np.std(item) for item in runtimes_per_solver]

        new_dict = dict(cluster, **{
            'runtimes_mean': runtimes_means,
            'runtime_variance': runtimes_variance,
            'runtimes_standard_deviation': runtimes_standard_deviation
        })

        finished_cluster_data.append(new_dict)

    return finished_cluster_data


