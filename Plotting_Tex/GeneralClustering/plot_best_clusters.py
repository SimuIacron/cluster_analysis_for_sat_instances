import itertools
from run_plotting import plot_best_cluster_comparison, plot_cluster_best_solver_distribution_relative
from run_plotting_histograms import plot_histograms_clustering, plot_boxplot_best_cluster, plot_histogram_best_cluster

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

# plot_best_cluster_comparison(['clustering_general/clustering_general_par2'],
#                              '',
#                              0, ['selected_data'],
#                              [output_merged[1:]],
#                              ['base', 'gate', 'runtimes', 'base gate', 'base runtimes', 'gate runtimes',
#                               'base gate runtimes'],
#                              100, 100, 20,
#                              output_file='/general_clustering/single_clusters/clustering_best_cluster_all_5000',
#                              show_plot=False,
#                              use_mat_plot=True)

plot_boxplot_best_cluster(input_file,
                          0, ['selected_data'],
                          [output_merged[1:]],
                          ['base', 'gate', 'runtimes', 'base gate', 'base runtimes', 'gate runtimes',
                           'base gate runtimes'],
                          max_cluster_amount=10000, min_cluster_size=20, angle=0,
                          output_file=output_directory + '/single_clusters/box_best_clusters_all_comb_20_10000',
                          show_plot=False)

plot_histogram_best_cluster(input_file,
                            0, ['selected_data'],
                            [output_merged[1:]],
                            ['base', 'gate', 'runtimes', 'base gate', 'base runtimes',
                             'gate runtimes',
                             'base gate runtimes'],
                            max_cluster_amount=10000, min_cluster_size=20, columns=3, bin_step=100,
                            output_file=output_directory + '/single_clusters/hist_best_clusters_all_comb_20_10000',
                            show_plot=False, normalize=False)

plot_boxplot_best_cluster(input_file,
                          0, ['selected_data'],
                          [output_merged[1:]],
                          ['base', 'gate', 'runtimes', 'base gate', 'base runtimes', 'gate runtimes',
                           'base gate runtimes'],
                          max_cluster_amount=10000, min_cluster_size=10, angle=0,
                          output_file=output_directory + '/single_clusters/box_best_clusters_all_comb_10_10000',
                          show_plot=False)

plot_histogram_best_cluster(input_file,
                            0, ['selected_data'],
                            [output_merged[1:]],
                            ['base', 'gate', 'runtimes', 'base gate', 'base runtimes',
                             'gate runtimes',
                             'base gate runtimes'],
                            max_cluster_amount=10000, min_cluster_size=10, columns=3, bin_step=100,
                            output_file=output_directory + '/single_clusters/hist_best_clusters_all_comb_10_10000',
                            show_plot=False, normalize=False)

plot_boxplot_best_cluster(input_file,
                          0, ['cluster_algorithm', 'selected_data'],
                          [['KMEANS', 'AFFINITY', 'MEANSHIFT', 'SPECTRAL', 'AGGLOMERATIVE', 'OPTICS', 'GAUSSIAN',
                          'DBSCAN',
                          'BIRCH'], output_merged[1:]],
                          ['K-Means', 'Affintiy Propagation', 'Meanshift', 'Spectral Clustering', 'Agglomerative',
                          'OPTICS',
                          'Gaussian', 'DBSCAN', 'BIRCH'],
                          max_cluster_amount=10000, min_cluster_size=10, angle=0,
                          output_file=output_directory + '/single_clusters/box_best_clusters_all_algo_10_10000',
                          show_plot=False)

input_dbs = [DatabaseReader.FEATURES_BASE, DatabaseReader.FEATURES_GATE]
output = sum([list(map(list, itertools.combinations(input_dbs, i))) for i in range(len(input_dbs) + 1)], [])
output_merged = []
for combination in output:
    comb = []
    for elem in combination:
        comb = comb + elem
    output_merged.append(comb)

plot_boxplot_best_cluster(input_file,
                          0, ['cluster_algorithm', 'selected_data'],
                          [['KMEANS', 'AFFINITY', 'MEANSHIFT', 'SPECTRAL', 'AGGLOMERATIVE', 'OPTICS', 'GAUSSIAN',
                          'DBSCAN',
                          'BIRCH'], output_merged[1:]],
                          ['K-Means', 'Affintiy Propagation', 'Meanshift', 'Spectral Clustering', 'Agglomerative',
                          'OPTICS',
                          'Gaussian', 'DBSCAN', 'BIRCH'],
                          max_cluster_amount=10000, min_cluster_size=10, angle=0,
                          output_file=output_directory + '/single_clusters/box_best_clusters_base_gate_algo_10_10000',
                          show_plot=False)


#
# plot_best_cluster_comparison(['clustering_general/clustering_general_par2'],
#                              '',
#                              0, ['selected_data'],
#                              [output_merged[1:]],
#                              ['base', 'gate', 'base gate'],
#                              100, 100, 100,
#                              output_file='/general_clustering/single_clusters/clustering_best_cluster_base_gate_comb_100',
#                              show_plot=False,
#                              use_mat_plot=True)
#
# plot_best_cluster_comparison(['clustering_general/clustering_general_par2'],
#                              '',
#                              0, ['cluster_algorithm', 'selected_data'],
#                              [['KMEANS', 'AFFINITY', 'MEANSHIFT', 'SPECTRAL', 'AGGLOMERATIVE', 'OPTICS', 'GAUSSIAN',
#                                'DBSCAN',
#                                'BIRCH'], output_merged[1:]],
#                              ['K-Means', 'Affintiy Propagation', 'Meanshift', 'Spectral Clustering', 'Agglomerative',
#                               'OPTICS',
#                               'Gaussian', 'DBSCAN', 'BIRCH'],
#                              100, 100, 100,
#                              output_file='/general_clustering/single_clusters/clustering_best_cluster_base_gate_algo_100',
#                              show_plot=False,
#                              use_mat_plot=True)
