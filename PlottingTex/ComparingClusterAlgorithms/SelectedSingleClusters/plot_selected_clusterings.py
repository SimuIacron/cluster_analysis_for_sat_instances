import numpy as np
from PlottingAndEvaluationFunctions.func_plot_single_clustering import plot_family_distribution_of_clusters, \
    plot_runtime_comparison_sbs
from PlottingAndEvaluationFunctions.func_stochastic_cluster_values import calculate_cluster_performance_score, \
    calculate_biggest_family_for_cluster, \
    calculate_feature_stochastic, \
    find_base_and_gate_features_with_low_std, check_performance_for_all_instances_of_major_family, \
    check_performance_for_instances_with_similar_feature_values, filter_specific_clustering, calculate_clusters_in_strip, \
    get_unsolvable_instances_amount, generate_csv_cluster_strip, count_instances_in_clusters_with_sbs, \
    generate_csv_csbs_csbss, filter_by_cluster_size
from DataFormats.DbInstance import DbInstance
from run_experiments import read_json
from PlottingTex.Preprocessing.spar2_score_visualisation import visualisation_spar2
from UtilScripts.util import get_combinations_of_databases

version = '6'
input_file_cluster = 'clustering_general_v{ver}/standardscaler/general_clustering_{ver}_clusters'.format(ver=version)
input_file_clustering = 'clustering_general_v{ver}/standardscaler/general_clustering_{ver}'.format(ver=version)
input_file_dataset = 'clustering_general_v{ver}/stochastic_values_dataset'.format(ver=version)
input_file_sbs = 'clustering_general_v{ver}/sbs_{ver}'.format(ver=version)
output_file = '/general_clustering_{ver}/specific_clustering/'.format(ver=version)

clusterings = [
    (33, 'kmeans_33/kmeans_33'),
    (25522, 'dbscan_25522/dbscan_25522'),
]
setting = 0
dpi = 120

output_merged, features = get_combinations_of_databases()

db_instance = DbInstance(features)

# write_json(input_file_dataset, calculate_stochastic_value_for_dataset(db_instance))

data_clustering = read_json(input_file_clustering)
data_clusters = read_json(input_file_cluster)
data_dataset = read_json(input_file_dataset)
sbs = read_json(input_file_sbs)
sbs_solver = sbs[1]["0"][0][0][0]

print('starting with {a} clusters'.format(a=len(data_clusters)))

filtered1 = filter_specific_clustering(data_clusters, clusterings[setting][0])

calculated = calculate_feature_stochastic(data_clustering, filtered1, data_dataset, db_instance)

deviation_data = calculate_cluster_performance_score(data_clustering, calculated, db_instance)
# sbs_data = calculate_factor_of_sbs_and_deviation_solver(data_clustering, deviation_data, sbs_solver, db_instance)
biggest_families = calculate_biggest_family_for_cluster(data_clustering, deviation_data, db_instance)
# best_clusters = filter_pareto_optimal_clusters(biggest_families,
#                                                ['cluster_performance_score',
#                                                 'family_total_percentage'],
#                                                [True, False])
interesting_features = find_base_and_gate_features_with_low_std(biggest_families, data_dataset, db_instance,
                                                                max_std=0.0001)
family_performance = check_performance_for_all_instances_of_major_family(data_clustering, interesting_features,
                                                                         db_instance)
# pareto_solvers = calculate_pareto_optimal_solvers_std_mean(family_performance, db_instance)
similar_instances_performance = check_performance_for_instances_with_similar_feature_values(family_performance,
                                                                                            data_clustering,
                                                                                            db_instance)

strip_par2 = calculate_clusters_in_strip(data_clustering, similar_instances_performance, db_instance)
unsolvable = get_unsolvable_instances_amount(data_clustering, strip_par2, db_instance)

# best_families = filter_best_cluster_for_each_family(family_performance, 'cluster_performance_score', minimize=True)
# clusterings = find_best_clustering_by_deviation_score(data_clustering, deviation_data)
sort = sorted(unsolvable, key=lambda d: d['cluster_size'], reverse=True)

quantile = np.quantile([cluster['cluster_size'] for cluster in sort], 0.25)
print('0.25-Quantile {a}'.format(a=quantile))

sort2 = []
outlier_cluster = []
for elem in sort:
    if elem['cluster_idx'] != -1:
        sort2.append(elem)
    else:
        outlier_cluster = [elem]
sort = outlier_cluster + sort2

filtered_size = filter_by_cluster_size(sort, quantile + 1)
print(count_instances_in_clusters_with_sbs(filtered_size, sbs_solver))

generate_csv_csbs_csbss(filtered_size, ['Cluster', 'CSBS', 'CSBSS'], output_file + clusterings[setting][1] + '_solvers')
generate_csv_cluster_strip(filtered_size, ['Cluster', 'Strip'], output_file + clusterings[setting][1] + '_strip')

excluded = ['levels_min'
                 'levels_none_mean', 'levels_none_variance', 'levels_none_min', 'levels_none_max',
                 'levels_none_entropy',
                 'horn_vars_min',
                 'inv_horn_vars_min',
                 'balance_vars_mean', 'balance_vars_variance', 'balance_vars_min', 'balance_vars_max',
                 'balance_vars_entropy', 'vcg_vdegrees_min',
                 'vg_degrees_min'
                 ]

# all_strip_solvers = []
# for cluster in sort:
#     current_strip_solvers = []
#     for item in cluster['par2_strip']:
#         if item[1] != DatabaseReader.TIMEOUT * 2:
#             current_strip_solvers.append(item[0])
#     all_strip_solvers = all_strip_solvers + current_strip_solvers
#
# count = Counter(all_strip_solvers)
#
# cluster_list = []
# for i, cluster in enumerate(sort):
#     for item in cluster['par2_strip']:
#         if item[1] != DatabaseReader.TIMEOUT * 2 and (item[0] == 'glucose' or item[0] == 'candy' or
#                                                       item[0] == 'glucose_var_decay099' or item[0] == 'lingeling'):
#             cluster_list.append((item[0], i))

cluster_index = 17
# boxplot_cluster_feature_distribution(sort[cluster_index], db_instance, dpi=dpi, use_base=True, use_gate=False,
#                                      angle=90, output_file=output_file + clusterings[setting][1] + '_' +
#                                                            str(cluster_index) + '_base',
#                                      show_plot=False, exclude_features=excluded)

pass
plot_family_distribution_of_clusters(filtered_size, db_instance, show_plot=False,
                                     output_file=output_file + clusterings[setting][1] + '_family_distribution',
                                     dpi=dpi)


# plot_performance_mean_std_and_performance_of_cluster(sort, db_instance, show_plot=False,
#                                                      output_file=output_file + clusterings[setting][
#                                                          1] + '_performance_score', dpi=dpi)
plot_runtime_comparison_sbs(filtered_size, sbs_solver, show_plot=False,
                            output_file=output_file + clusterings[setting][1] + '_sbs_comparison', dpi=dpi)
# boxplot_runtimes_distribution_per_cluster(sort, db_instance,
#                                           output_file=output_file + clusterings[setting][1] + '_boxplot_runtimes',
#                                           dpi=dpi)
# plot_biggest_family_runtime(sort, show_plot=False, output_file=output_file + clusterings[setting][1] + '_par2_scores',
#                             dpi=dpi)
#
# for i, cluster in enumerate(sort):
#     boxplot_cluster_feature_distribution(cluster, db_instance, dpi=dpi, use_base=True, use_gate=False,
#                                          angle=90, output_file=output_file + clusterings[setting][1] + '_' +
#                                                                str(i) + '_base',
#                                          show_plot=False)

index_list = [[12, 13, 18], [4,6,17,24,26]]

for index in index_list[setting]:
    cluster = filtered_size[index]
    visualisation_spar2(data_clustering, cluster, db_instance,show_plot=False,
                                output_file=output_file + clusterings[setting][1] + '_spar_vis_' + str(index), dpi=dpi)