import itertools
from run_plotting import plot_cbs_comparison
from run_plotting_histograms import plot_histograms_clustering, plot_boxplot_clustering

from util_scripts import DatabaseReader

output_directory = '/general_clustering_2'
input_file = 'clustering_general_v2/general_clustering_2_par2'
sbs_file = 'vbs_sbs/sbs'

# All features ---------------------------------------------------------------------------------------------------------

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

# plot_cbs_comparison([dir + '/clustering_general_par2'], 'vbs_sbs/vbs', 'vbs_sbs/sbs',
#                       '',
#                       0, ['selected_data'],
#                       [output_merged[1:]],
#                       ['base', 'gate', 'runtimes', 'base gate', 'base runtimes', 'gate runtimes', 'base gate runtimes'],
#                       20, 200, output_file=dir + '/clustering_default_comb_all_dbs_new',
#                       show_plot=False,
#                       use_mat_plot=True, use_dash_plot=False, show_complete_legend=True)
#
# plot_cbs_comparison([dir + '/clustering_general_par2'], 'vbs_sbs/vbs', 'vbs_sbs/sbs',
#                       '',
#                       0, ['cluster_algorithm', 'selected_data'],
#                       [['KMEANS', 'AFFINITY', 'MEANSHIFT', 'SPECTRAL', 'AGGLOMERATIVE', 'OPTICS', 'GAUSSIAN', 'DBSCAN',
#                         'BIRCH'], output_merged[1:]],
#                       ['K-Means', 'Affintiy Propagation', 'Meanshift', 'Spectral Clustering', 'Agglomerative', 'OPTICS',
#                        'Gaussian', 'DBSCAN', 'BIRCH'],
#                       20, 200, output_file=dir + '/clustering_default_algo_all_dbs_new',
#                       show_plot=False,
#                       use_mat_plot=True, use_dash_plot=False, show_complete_legend=True)


plot_histograms_clustering(input_file, sbs_file,
                           0, ['selected_data'],
                           [output_merged[1:]],
                           ['base', 'gate', 'runtimes', 'base gate', 'base runtimes', 'gate runtimes',
                            'base gate runtimes'],
                           max_cluster_amount=20, columns=3,
                           bin_step=10, height=0.11, output_file=output_directory + '/hist_clustering_default_comb_all',
                           normalize=True, dpi=192)

plot_boxplot_clustering(input_file,
                        0, ['selected_data'],
                        [output_merged[1:]],
                        ['base', 'gate', 'runtimes', 'base gate', 'base runtimes', 'gate runtimes',
                         'base gate runtimes'],
                        max_cluster_amount=20, output_file=output_directory + '/box_clustering_default_comb_all',
                        sbs_file=sbs_file, dpi=192, angle=20)

plot_histograms_clustering(input_file, sbs_file,
                           0, ['cluster_algorithm', 'selected_data'],
                           [['KMEANS', 'AFFINITY', 'MEANSHIFT', 'SPECTRAL', 'AGGLOMERATIVE', 'OPTICS', 'GAUSSIAN',
                             'DBSCAN',
                             'BIRCH'], output_merged[1:]],
                           ['K-Means', 'Affintiy Propagation', 'Meanshift', 'Spectral Clustering', 'Agglomerative',
                            'OPTICS',
                            'Gaussian', 'DBSCAN', 'BIRCH'],
                           max_cluster_amount=20, columns=3,
                           bin_step=10, height=0.11, output_file=output_directory + '/hist_clustering_default_algo_all',
                           normalize=True, dpi=192)

plot_boxplot_clustering(input_file,
                        0, ['cluster_algorithm', 'selected_data'],
                        [['KMEANS', 'AFFINITY', 'MEANSHIFT', 'SPECTRAL', 'AGGLOMERATIVE', 'OPTICS', 'GAUSSIAN',
                          'DBSCAN',
                          'BIRCH'], output_merged[1:]],
                        ['K-Means', 'Affintiy Propagation', 'Meanshift', 'Spectral Clustering', 'Agglomerative',
                         'OPTICS',
                         'Gaussian', 'DBSCAN', 'BIRCH'],
                        max_cluster_amount=20, output_file=output_directory + '/box_clustering_default_algo_all',
                        sbs_file=sbs_file, dpi=192, angle=20)

# Only base and gate ---------------------------------------------------------------------------------------------------

input_dbs = [DatabaseReader.FEATURES_BASE, DatabaseReader.FEATURES_GATE]
output = sum([list(map(list, itertools.combinations(input_dbs, i))) for i in range(len(input_dbs) + 1)], [])
output_merged = []
for combination in output:
    comb = []
    for elem in combination:
        comb = comb + elem
    output_merged.append(comb)

plot_histograms_clustering(input_file, sbs_file,
                           0, ['cluster_algorithm', 'selected_data'],
                           [['KMEANS', 'AFFINITY', 'MEANSHIFT', 'SPECTRAL', 'AGGLOMERATIVE', 'OPTICS', 'GAUSSIAN',
                             'DBSCAN',
                             'BIRCH'], output_merged[1:]],
                           ['K-Means', 'Affintiy Propagation', 'Meanshift', 'Spectral Clustering', 'Agglomerative',
                            'OPTICS',
                            'Gaussian', 'DBSCAN', 'BIRCH'],
                           max_cluster_amount=20, columns=3,
                           bin_step=10, height=0.11,
                           output_file=output_directory + '/hist_clustering_default_algo_base_gate',
                           normalize=True, dpi=192)

plot_boxplot_clustering(input_file,
                        0, ['cluster_algorithm', 'selected_data'],
                        [['KMEANS', 'AFFINITY', 'MEANSHIFT', 'SPECTRAL', 'AGGLOMERATIVE', 'OPTICS', 'GAUSSIAN',
                          'DBSCAN',
                          'BIRCH'], output_merged[1:]],
                        ['K-Means', 'Affintiy Propagation', 'Meanshift', 'Spectral Clustering', 'Agglomerative',
                         'OPTICS',
                         'Gaussian', 'DBSCAN', 'BIRCH'],
                        max_cluster_amount=20,
                        output_file=output_directory + '/box_clustering_default_algo_base_gate', sbs_file=sbs_file,
                        dpi=192, angle=20)

# plot_cbs_comparison([dir + '/clustering_general_par2'], 'vbs_sbs/vbs', 'vbs_sbs/sbs',
#                       '',
#                     0, ['selected_data'],
#                     [output_merged[1:]],
#                     ['base', 'gate', 'base gate'],
#                     20, 200, output_file=dir + '/clustering_default_comb_base_gate_new',
#                     show_plot=False,
#                     use_mat_plot=True, use_dash_plot=False, show_complete_legend=True)
#
# plot_cbs_comparison([dir + '/clustering_general_par2'], 'vbs_sbs/vbs', 'vbs_sbs/sbs',
#                       '',
#                       0, ['cluster_algorithm', 'selected_data'],
#                       [['KMEANS', 'AFFINITY', 'MEANSHIFT', 'SPECTRAL', 'AGGLOMERATIVE', 'OPTICS', 'GAUSSIAN', 'DBSCAN',
#                         'BIRCH'], output_merged[1:]],
#                       ['K-Means', 'Affintiy Propagation', 'Meanshift', 'Spectral Clustering', 'Agglomerative', 'OPTICS',
#                        'Gaussian', 'DBSCAN', 'BIRCH'],
#                       20, 200, output_file=dir + '/clustering_default_algo_base_gate_new',
#                       show_plot=False,
#                       use_mat_plot=True, use_dash_plot=False, show_complete_legend=True)
#
