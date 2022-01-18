import itertools

from ClusterAnalysis.stochastic_cluster_values import export_variance_mean_of_cluster
from DataFormats.DbInstance import DbInstance
from run_experiments import read_json
from run_plotting_clusters import export_clusters_sorted_best, compare_with_family, plot_biggest_cluster_for_family
from util_scripts import DatabaseReader

input_file_cluster = 'clustering_general_v2/single_clusters/clustering_general_v2_best_cluster'
input_file_clustering = 'clustering_general_v2/general_clustering_2_par2'
output_dir_stochastic = 'clustering_general_v2/clustering_general_clusters_stochastic'

temp_solver_features = DatabaseReader.FEATURES_SOLVER.copy()
temp_solver_features.pop(14)
temp_solver_features.pop(7)
input_dbs = [DatabaseReader.FEATURES_BASE, DatabaseReader.FEATURES_GATE, temp_solver_features]
output = sum([list(map(list, itertools.combinations(input_dbs, i))) for i in range(len(input_dbs) + 1)], [])
output_merged = []
for combination in output:
    comb = []
    for elem in combination:
        comb = comb + elem
    output_merged.append(comb)

features = []
for feature_vector in input_dbs:
    features = features + feature_vector

db_instance = DbInstance(features)

export_variance_mean_of_cluster(input_file_clustering, input_file_cluster, output_dir_stochastic, db_instance)