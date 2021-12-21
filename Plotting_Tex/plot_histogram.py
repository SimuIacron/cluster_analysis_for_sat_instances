import itertools
from run_plotting import plot_cpar2_comparison
from run_plotting_histograms import plot_histograms_clustering

from util_scripts import DatabaseReader

dir1 = 'scaling_standardscaler'
dir2 = 'clustering_general'
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

# plot_histograms_clustering(dir1 + '/standardscaler_linearscaler_clustering_par2',
#                             0, ['scaling_algorithm'],
#                             [['SCALEMINUSPLUS1', 'STANDARDSCALER']],
#                             ['[-1,+1]', 'Standard Scaler'],
#                             max_cluster_amount=20, columns=2,
#                             bin_step=10, height=500, output_file='') # dir1 + '/hist_standardscaler_20')

# plot_histograms_clustering(dir2 + '/clustering_general_par2',
#                            0, ['selected_data'],
#                            [output_merged[1:]],
#                            ['base', 'gate', 'runtimes', 'base gate', 'base runtimes', 'gate runtimes',
#                             'base gate runtimes'],
#                            max_cluster_amount=20, columns=4,
#                            bin_step=10, height=800, output_file=dir2 + '/hist_cluster')

# plot_histograms_clustering(dir2 + '/clustering_general_par2',
#                            0, ['cluster_algorithm'],
#                            [['KMEANS', 'AFFINITY', 'MEANSHIFT', 'SPECTRAL', 'AGGLOMERATIVE', 'OPTICS', 'GAUSSIAN',
#                              'DBSCAN',
#                              'BIRCH']],
#                            ['K-Means', 'Affintiy Propagation', 'Meanshift', 'Spectral Clustering', 'Agglomerative',
#                             'OPTICS',
#                             'Gaussian', 'DBSCAN', 'BIRCH'],
#                            max_cluster_amount=20, columns=3,
#                            bin_step=10, height=0.11, output_file=dir2 + '/hist_cluster_algo', normalize=True)

input_dbs = [DatabaseReader.FEATURES_BASE, DatabaseReader.FEATURES_GATE]
output = sum([list(map(list, itertools.combinations(input_dbs, i))) for i in range(len(input_dbs) + 1)], [])
output_merged = []
for combination in output:
    comb = []
    for elem in combination:
        comb = comb + elem
    output_merged.append(comb)

plot_histograms_clustering(dir2 + '/clustering_general_par2',
                           0, ['cluster_algorithm', 'selected_data'],
                           [['KMEANS', 'AFFINITY', 'MEANSHIFT', 'SPECTRAL', 'AGGLOMERATIVE', 'OPTICS', 'GAUSSIAN',
                             'DBSCAN',
                             'BIRCH'], output_merged[1:]],
                           ['K-Means', 'Affintiy Propagation', 'Meanshift', 'Spectral Clustering', 'Agglomerative',
                            'OPTICS',
                            'Gaussian', 'DBSCAN', 'BIRCH'],
                           max_cluster_amount=20, columns=3,
                           bin_step=10, height=0.11, output_file=dir2 + '/hist_cluster_algo', normalize=True)
