from PlottingAndEvaluationFunctions.func_plot_clustering_algorithm_comparison import plot_histograms_clustering, plot_boxplot_clustering
from util_scripts.util import get_combinations_of_databases

version = '6'
output_directory = '/general_clustering_{ver}'.format(ver=version)
input_file = 'clustering_general_v{ver}/standardscaler/general_clustering_{ver}_par2'.format(ver=version)
sbs_file = 'clustering_general_v{ver}/sbs_{ver}'.format(ver=version)
vbs_file = 'clustering_general_v{ver}/vbs_{ver}'.format(ver=version)

max_cluster_amounts = 35
remove_from_sbs = 25

# All features ---------------------------------------------------------------------------------------------------------


output_merged, features = get_combinations_of_databases()

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
                           [output_merged],
                           ['base', 'gate', 'runtimes', 'base gate', 'base runtimes', 'gate runtimes',
                            'base gate runtimes'],
                           max_cluster_amount=max_cluster_amounts, columns=3,
                           bin_step=10, height=0.11, output_file=output_directory + '/hist_clustering_default_comb_all',
                           normalize=True, dpi=192)

plot_boxplot_clustering(input_file,
                        0, ['selected_data'],
                        [output_merged],
                        ['base', 'gate', 'runtimes', 'base gate', 'base runtimes', 'gate runtimes',
                         'base gate runtimes'],
                        max_cluster_amount=max_cluster_amounts, output_file=output_directory + '/box_clustering_default_comb_all',
                        sbs_file=sbs_file, dpi=192, angle=20)

plot_histograms_clustering(input_file, sbs_file,
                           0, ['cluster_algorithm', 'selected_data'],
                           [['KMEANS', 'AFFINITY', 'MEANSHIFT', 'SPECTRAL', 'AGGLOMERATIVE', 'OPTICS', 'GAUSSIAN',
                             'DBSCAN',
                             'BIRCH'], output_merged],
                           ['K-Means', 'Affintiy Propagation', 'Meanshift', 'Spectral Clustering', 'Agglomerative',
                            'OPTICS',
                            'Gaussian', 'DBSCAN', 'BIRCH'],
                           max_cluster_amount=max_cluster_amounts, columns=3,
                           bin_step=10, height=0.11, output_file=output_directory + '/hist_clustering_default_algo_all',
                           normalize=True, dpi=192)

plot_boxplot_clustering(input_file,
                        0, ['cluster_algorithm', 'selected_data'],
                        [['KMEANS', 'AFFINITY', 'MEANSHIFT', 'SPECTRAL', 'AGGLOMERATIVE', 'OPTICS', 'GAUSSIAN',
                          'DBSCAN',
                          'BIRCH'], output_merged],
                        ['K-Means', 'Affintiy Propagation', 'Meanshift', 'Spectral Clustering', 'Agglomerative',
                         'OPTICS',
                         'Gaussian', 'DBSCAN', 'BIRCH'],
                        max_cluster_amount=max_cluster_amounts, output_file=output_directory + '/box_clustering_default_algo_all',
                        sbs_file=sbs_file, dpi=192, angle=20, remove_box_if_all_values_in_range_of_sbs=remove_from_sbs)

# Only base and gate ---------------------------------------------------------------------------------------------------

output_merged, features_2 = get_combinations_of_databases(use_solver=False)

plot_histograms_clustering(input_file, sbs_file,
                           0, ['cluster_algorithm', 'selected_data'],
                           [['KMEANS', 'AFFINITY', 'MEANSHIFT', 'SPECTRAL', 'AGGLOMERATIVE', 'OPTICS', 'GAUSSIAN',
                             'DBSCAN',
                             'BIRCH'], output_merged],
                           ['K-Means', 'Affintiy Propagation', 'Meanshift', 'Spectral Clustering', 'Agglomerative',
                            'OPTICS',
                            'Gaussian', 'DBSCAN', 'BIRCH'],
                           max_cluster_amount=max_cluster_amounts, columns=3,
                           bin_step=10, height=0.11,
                           output_file=output_directory + '/hist_clustering_default_algo_base_gate',
                           normalize=True, dpi=192)

plot_boxplot_clustering(input_file,
                        0, ['cluster_algorithm', 'selected_data'],
                        [['KMEANS', 'AFFINITY', 'MEANSHIFT', 'SPECTRAL', 'AGGLOMERATIVE', 'OPTICS', 'GAUSSIAN',
                          'DBSCAN',
                          'BIRCH'], output_merged],
                        ['K-Means', 'Affintiy Propagation', 'Meanshift', 'Spectral Clustering', 'Agglomerative',
                         'OPTICS',
                         'Gaussian', 'DBSCAN', 'BIRCH'],
                        max_cluster_amount=max_cluster_amounts,
                        output_file=output_directory + '/box_clustering_default_algo_base_gate', sbs_file=sbs_file,
                        dpi=192, angle=20, remove_box_if_all_values_in_range_of_sbs=remove_from_sbs)

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
