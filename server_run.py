import itertools

from ClusterAnalysis.plot_single_clustering import plot_family_distribution_of_clusters, plot_runtime_comparison_sbs, \
    plot_performance_mean_std_and_performance_of_cluster, boxplot_runtimes_distribution_per_cluster
from ClusterAnalysis.plot_single_clusters import boxplot_cluster_feature_distribution, barchart_compare_runtime_scores, \
    barchart_compare_sbs_speedup
from ClusterAnalysis.stochastic_cluster_values import export_variance_mean_of_cluster, filter_cluster_data, \
    calculate_cluster_performance_score, calculate_biggest_family_for_cluster, calculate_pareto_optimal_solvers_std_mean, \
    filter_best_cluster_for_each_family, filter_pareto_optimal_clusters, calculate_feature_stochastic, \
    find_base_and_gate_features_with_low_std, sort_after_param, check_performance_for_all_instances_of_major_family, \
    check_performance_for_instances_with_similar_feature_values, filter_non_clusters, filter_same_cluster, \
    calculate_factor_of_sbs_and_deviation_solver, search_clusters_with_unsolvable_instances, \
    find_best_clustering_by_performance_score, filter_specific_clustering
from ClusterAnalysis.stochastic_values_dataset import calculate_stochastic_value_for_dataset
from DataFormats.DbInstance import DbInstance
from run_experiments import read_json, write_json
from run_plotting_clusters import export_clusters_sorted_best, compare_with_family, plot_biggest_cluster_for_family
from util_scripts import DatabaseReader

input_file_cluster = 'clustering_general_v3/single_clusters/general_clustering_3_clusters'
input_file_clustering = 'clustering_general_v3/general_clustering_3'
# output_dir_stochastic = 'clustering_general_v3/single_clusters/clustering_general_clusters_stochastic'
input_file_dataset = 'clustering_general_v3/stochastic_values_dataset'
input_file_sbs = 'clustering_general_v3/sbs_3'
output_file = '/general_clustering_3/specific_clustering/kmeans/'

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

# values = calculate_stochastic_value_for_dataset(db_instance)
# write_json(input_file_dataset, values)

# export_variance_mean_of_cluster(input_file_clustering, input_file_cluster, output_dir_stochastic, db_instance)

data_clustering = read_json(input_file_clustering)
data_clusters = read_json(input_file_cluster)
data_dataset = read_json(input_file_dataset)
sbs = read_json(input_file_sbs)
sbs_solver = sbs[1]["0"][0][0][0]

print('starting with {a} clusters'.format(a=len(data_clusters)))

# filtered1 = filter_cluster_data(data_clustering, data_clusters,
#                                 ['selected_data', 'cluster_algorithm'],
#                                 [[DatabaseReader.FEATURES_BASE, DatabaseReader.FEATURES_GATE,
#                                   DatabaseReader.FEATURES_BASE + DatabaseReader.FEATURES_GATE],
#                                  ['AGGLOMERATIVE']],
#                                 #'KMEANS', 'DBSCAN', 'AGGLOMERATIVE'
#                                 (0, 10000), (5, 30))

filtered1 = filter_specific_clustering(data_clusters, 8)

# filtered2 = filter_non_clusters(filtered1)
# filtered3 = filter_same_cluster(data_clustering, filtered2)
calculated = calculate_feature_stochastic(data_clustering, filtered1, data_dataset, db_instance)

deviation_data = calculate_cluster_performance_score(calculated, db_instance)
# sbs_data = calculate_factor_of_sbs_and_deviation_solver(data_clustering, deviation_data, sbs_solver, db_instance)
biggest_families = calculate_biggest_family_for_cluster(data_clustering, deviation_data, db_instance)
# best_clusters = filter_pareto_optimal_clusters(biggest_families,
#                                                ['cluster_performance_score',
#                                                 'family_total_percentage'],
#                                                [True, False])
interesting_features = find_base_and_gate_features_with_low_std(biggest_families, data_dataset, db_instance,
                                                                max_std=0.00001)
family_performance = check_performance_for_all_instances_of_major_family(interesting_features, db_instance)
# pareto_solvers = calculate_pareto_optimal_solvers_std_mean(family_performance, db_instance)
similar_instances_performance = check_performance_for_instances_with_similar_feature_values(family_performance,
                                                                                            data_clustering,
                                                                                            db_instance)


# best_families = filter_best_cluster_for_each_family(family_performance, 'cluster_performance_score', minimize=True)
# clusterings = find_best_clustering_by_deviation_score(data_clustering, deviation_data)
sort = sort_after_param(similar_instances_performance, 'cluster_performance_score', descending=False)

pass
plot_family_distribution_of_clusters(sort, db_instance, show_plot=False, biggest_n_families=2, output_file=output_file + 'kmeans_family_distribution')
# plot_performance_mean_std_and_performance_of_cluster(sort, db_instance, show_plot=False, output_file=output_file + 'kmeans_performance_score')
# plot_runtime_comparison_sbs(sort, sbs_solver, show_plot=False, output_file=output_file + 'kmeans_sbs_comparison')
# boxplot_runtimes_distribution_per_cluster(sort, db_instance, output_file=output_file + 'kmeans_boxplot_runtimes')

# for cluster in sort:
#     boxplot_cluster_feature_distribution(cluster, db_instance, use_base=True, use_gate=False,
#                                          dpi=100, show_plot=False,
#                                          output_file=output_file + str(cluster['cluster_idx']) + '_base')
#     boxplot_cluster_feature_distribution(value, db_instance, use_base=False, use_gate=True,
#                                          dpi=100, show_plot=False,
#                                          output_file=output_file + 'feature_distribution/' + key + '_gate')

# family_list = []
# for key, item in best_families.items():
#     family_list.append(item)
# barchart_compare_runtime_scores(family_list, db_instance, output_file=output_file + 'family_cluster_scores', dpi=100)
#
# barchart_compare_runtime_scores(sort[:30], db_instance, output_file=output_file + 'best_cluster_scores', dpi=100)
# 
# family_list = []
# for key, item in best_families.items():
#     family_list.append(item)
# barchart_compare_sbs_speedup(family_list, output_file=output_file + 'family_sbs_speedup', dpi=100)
#
# barchart_compare_sbs_speedup(sort[:30], output_file=output_file + 'best_sbs_speedup', dpi=100)
