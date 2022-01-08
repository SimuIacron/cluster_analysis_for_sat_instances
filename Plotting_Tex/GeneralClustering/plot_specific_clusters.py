import itertools

from DataFormats.DbInstance import DbInstance
from run_plotting import plot_par2, plot_single_cluster_distribution_family, plot_cluster_best_solver_distribution, \
    plot_cluster_best_solver_distribution_relative
from util_scripts import DatabaseReader

output_directory = '/general_clustering_2'
input_file = 'clustering_general_v2/general_clustering_2_par2'

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

# plot_single_cluster_distribution_family('clustering_general/clustering_general_par2', db_instance,
#                                  'best_clusters/clustering_best_cluster_base_gate_comb_100',
#                                  50, 10, 'best_clusters/specific_clusters/best_100_family', show_plot=True, use_mat_plot=True)

# plot_cluster_best_solver_distribution('clustering_general/clustering_general_par2', 1058, 3, db_instance,
#                                       output_file='best_clusters/specific_clusters/1058_3_solver', show_plot=False)
#
# plot_cluster_best_solver_distribution('clustering_general/clustering_general_par2', 184, 4, db_instance,
#                                       output_file='best_clusters/specific_clusters/184_4_solver', show_plot=False)
#
# plot_cluster_best_solver_distribution('clustering_general/clustering_general_par2', 8, 5, db_instance,
#                                       output_file='best_clusters/specific_clusters/8_5_solver', show_plot=False)
#
# plot_cluster_best_solver_distribution('clustering_general/clustering_general_par2', 1058, 0, db_instance,
#                                       output_file='best_clusters/specific_clusters/1058_0_solver', show_plot=False)

# plot_cluster_best_solver_distribution_relative('clustering_general/clustering_general_par2', 1058, 3, db_instance,
#                                       output_file='best_clusters/specific_clusters/1058_3_solver_relative_025', show_plot=False)
# 
# plot_cluster_best_solver_distribution_relative('clustering_general/clustering_general_par2', 184, 4, db_instance,
#                                       output_file='best_clusters/specific_clusters/184_4_solver_relative_025', show_plot=False)
#
# plot_cluster_best_solver_distribution_relative('clustering_general/clustering_general_par2', 8, 5, db_instance,
#                                       output_file='best_clusters/specific_clusters/8_5_solver_relative_025', show_plot=False)
#
# plot_cluster_best_solver_distribution_relative('clustering_general/clustering_general_par2', 1058, 0, db_instance,
#                                       output_file='best_clusters/specific_clusters/1058_0_solver_relative_025', show_plot=False)
#
# plot_cluster_best_solver_distribution_relative('clustering_general/clustering_general_par2', 1058, 3, db_instance,
#                                                quantile_value=0.1,
#                                       output_file='best_clusters/specific_clusters/1058_3_solver_relative_010', show_plot=False)
#
# plot_cluster_best_solver_distribution_relative('clustering_general/clustering_general_par2', 184, 4, db_instance,
#                                                quantile_value=0.1,
#                                       output_file='best_clusters/specific_clusters/184_4_solver_relative_010', show_plot=False)
#
# plot_cluster_best_solver_distribution_relative('clustering_general/clustering_general_par2', 8, 5, db_instance,
#                                                quantile_value=0.1,
#                                       output_file='best_clusters/specific_clusters/8_5_solver_relative_010', show_plot=False)
#
# plot_cluster_best_solver_distribution_relative('clustering_general/clustering_general_par2', 1058, 0, db_instance,
#                                                quantile_value=0.1,
#                                       output_file='best_clusters/specific_clusters/1058_0_solver_relative_010', show_plot=False)

# plot_cluster_best_solver_distribution_relative('clustering_general/clustering_general_par2', 1058, 3, db_instance,
#                                                quantile_value=0.25, scale_with_log=2,
#                                       output_file='best_clusters/specific_clusters/1058_3_solver_relative_025_log2', show_plot=False)
#
# plot_cluster_best_solver_distribution_relative('clustering_general/clustering_general_par2', 184, 4, db_instance,
#                                                quantile_value=0.25, scale_with_log=2,
#                                       output_file='best_clusters/specific_clusters/184_4_solver_relative_025_log2', show_plot=False)
#
# plot_cluster_best_solver_distribution_relative('clustering_general/clustering_general_par2', 8, 5, db_instance,
#                                                quantile_value=0.25, scale_with_log=2,
#                                       output_file='best_clusters/specific_clusters/8_5_solver_relative_025_log2', show_plot=False)
#
# plot_cluster_best_solver_distribution_relative('clustering_general/clustering_general_par2', 1058, 0, db_instance,
#                                                quantile_value=0.25, scale_with_log=2,
#                                       output_file='best_clusters/specific_clusters/1058_0_solver_relative_025_log2', show_plot=False)

plot_cluster_best_solver_distribution_relative(input_file, 6226, 10, db_instance,
                                               quantile_value=0.1, scale_with_log=2,
                                               output_file=output_directory + '/single_clusters/specific_clusters/1058_3_solver_relative_010_log2',
                                               show_plot=False)

# plot_cluster_best_solver_distribution_relative(input_file, 184, 4, db_instance,
#                                                quantile_value=0.1, scale_with_log=2,
#                                                output_file=output_directory + '/single_clusters/specific_clusters/184_4_solver_relative_010_log2',
#                                                show_plot=False)
#
# plot_cluster_best_solver_distribution_relative(input_file, 8, 5, db_instance,
#                                                quantile_value=0.1, scale_with_log=2,
#                                                output_file=output_directory + '/single_clusters/specific_clusters/8_5_solver_relative_010_log2',
#                                                show_plot=False)
#
# plot_cluster_best_solver_distribution_relative(input_file, 1058, 0, db_instance,
#                                                quantile_value=0.1, scale_with_log=2,
#                                                output_file=output_directory + '/single_clusters/specific_clusters/1058_0_solver_relative_010_log2',
#                                                show_plot=False)
