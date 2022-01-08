import itertools
from run_plotting import plot_cbs_comparison
from run_plotting_histograms import plot_histograms_clustering, plot_boxplot_clustering

from util_scripts import DatabaseReader

directory = 'clustering_general_cap_60'
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


plot_histograms_clustering(directory + '/general_clustering_cap_60_par2', sbs_file,
                           0, ['selected_data'],
                           [output_merged[1:]],
                           ['base', 'gate', 'runtimes', 'base gate', 'base runtimes', 'gate runtimes',
                            'base gate runtimes'],
                           max_cluster_amount=20, columns=3,
                           bin_step=10, height=0.11,
                           output_file='/general_clustering_cap_60/hist_clustering_default_cap_60_comb_all',
                           normalize=True)

plot_boxplot_clustering(directory + '/general_clustering_cap_60_par2',
                        0, ['selected_data'],
                        [output_merged[1:]],
                        ['base', 'gate', 'runtimes', 'base gate', 'base runtimes', 'gate runtimes',
                         'base gate runtimes'],
                        max_cluster_amount=20,
                        output_file='/general_clustering_cap_60/box_clustering_default_cap_60_comb_all')

plot_histograms_clustering(directory + '/general_clustering_cap_60_par2', sbs_file,
                           0, ['cluster_algorithm', 'selected_data'],
                           [['KMEANS', 'AFFINITY', 'MEANSHIFT', 'SPECTRAL', 'AGGLOMERATIVE', 'OPTICS', 'GAUSSIAN',
                             'DBSCAN',
                             'BIRCH'], output_merged[1:]],
                           ['K-Means', 'Affintiy Propagation', 'Meanshift', 'Spectral Clustering', 'Agglomerative',
                            'OPTICS',
                            'Gaussian', 'DBSCAN', 'BIRCH'],
                           max_cluster_amount=20, columns=3,
                           bin_step=10, height=0.11,
                           output_file='/general_clustering_cap_60/hist_clustering_default_cap_60_algo_all',
                           normalize=True)

plot_boxplot_clustering(directory + '/general_clustering_cap_60_par2',
                        0, ['cluster_algorithm', 'selected_data'],
                        [['KMEANS', 'AFFINITY', 'MEANSHIFT', 'SPECTRAL', 'AGGLOMERATIVE', 'OPTICS', 'GAUSSIAN',
                          'DBSCAN',
                          'BIRCH'], output_merged[1:]],
                        ['K-Means', 'Affintiy Propagation', 'Meanshift', 'Spectral Clustering', 'Agglomerative',
                         'OPTICS',
                         'Gaussian', 'DBSCAN', 'BIRCH'],
                        max_cluster_amount=20,
                        output_file='/general_clustering_cap_60/box_clustering_default_cap_60_algo_all')

# Only base and gate ---------------------------------------------------------------------------------------------------

input_dbs = [DatabaseReader.FEATURES_BASE, DatabaseReader.FEATURES_GATE]
output = sum([list(map(list, itertools.combinations(input_dbs, i))) for i in range(len(input_dbs) + 1)], [])
output_merged = []
for combination in output:
    comb = []
    for elem in combination:
        comb = comb + elem
    output_merged.append(comb)

plot_histograms_clustering(directory + '/general_clustering_cap_60_par2', sbs_file,
                           0, ['cluster_algorithm', 'selected_data'],
                           [['KMEANS', 'AFFINITY', 'MEANSHIFT', 'SPECTRAL', 'AGGLOMERATIVE', 'OPTICS', 'GAUSSIAN',
                             'DBSCAN',
                             'BIRCH'], output_merged[1:]],
                           ['K-Means', 'Affintiy Propagation', 'Meanshift', 'Spectral Clustering', 'Agglomerative',
                            'OPTICS',
                            'Gaussian', 'DBSCAN', 'BIRCH'],
                           max_cluster_amount=20, columns=3,
                           bin_step=10, height=0.11,
                           output_file='/general_clustering_cap_60/hist_clustering_default_cap_60_algo_base_gate',
                           normalize=True)

plot_boxplot_clustering(directory + '/general_clustering_cap_60_par2',
                        0, ['cluster_algorithm', 'selected_data'],
                        [['KMEANS', 'AFFINITY', 'MEANSHIFT', 'SPECTRAL', 'AGGLOMERATIVE', 'OPTICS', 'GAUSSIAN',
                          'DBSCAN',
                          'BIRCH'], output_merged[1:]],
                        ['K-Means', 'Affintiy Propagation', 'Meanshift', 'Spectral Clustering', 'Agglomerative',
                         'OPTICS',
                         'Gaussian', 'DBSCAN', 'BIRCH'],
                        max_cluster_amount=20,
                        output_file='/general_clustering_cap_60/box_clustering_default_cap_60_algo_base_gate')