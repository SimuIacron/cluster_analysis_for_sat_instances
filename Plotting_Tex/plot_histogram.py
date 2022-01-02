import itertools
from run_plotting import plot_cpar2_comparison
from run_plotting_histograms import plot_histograms_clustering, plot_boxplot_clustering

from util_scripts import DatabaseReader

dir1 = 'scaling_standardscaler'
dir2 = 'clustering_general'
dir3 = 'single_solver'
temp_solver_features = DatabaseReader.FEATURES_SOLVER.copy()
temp_solver_features.pop(14)
temp_solver_features.pop(7)
input_dbs = [DatabaseReader.FEATURES_BASE, DatabaseReader.FEATURES_GATE, temp_solver_features]
# output = sum([list(map(list, itertools.combinations(input_dbs, i))) for i in range(len(input_dbs) + 1)], [])
# output_merged = []
# for combination in output:
#     comb = []
#     for elem in combination:
#         comb = comb + elem
#     output_merged.append(comb)

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

# input_dbs = [DatabaseReader.FEATURES_BASE, DatabaseReader.FEATURES_GATE]
# output = sum([list(map(list, itertools.combinations(input_dbs, i))) for i in range(len(input_dbs) + 1)], [])
# output_merged = []
# for combination in output:
#     comb = []
#     for elem in combination:
#         comb = comb + elem
#     output_merged.append(comb)
#
# plot_histograms_clustering(dir2 + '/clustering_general_par2',
#                            0, ['cluster_algorithm', 'selected_data'],
#                            [['KMEANS', 'AFFINITY', 'MEANSHIFT', 'SPECTRAL', 'AGGLOMERATIVE', 'OPTICS', 'GAUSSIAN',
#                              'DBSCAN',
#                              'BIRCH'], output_merged[1:]],
#                            ['K-Means', 'Affintiy Propagation', 'Meanshift', 'Spectral Clustering', 'Agglomerative',
#                             'OPTICS',
#                             'Gaussian', 'DBSCAN', 'BIRCH'],
#                            max_cluster_amount=20, columns=3,
#                            bin_step=10, height=0.11, output_file=dir2 + '/hist_algo_vase_gate', normalize=True)

output = [[DatabaseReader.FEATURES_BASE], [DatabaseReader.FEATURES_GATE],
          [DatabaseReader.FEATURES_BASE, DatabaseReader.FEATURES_GATE],
          [DatabaseReader.FEATURES_BASE, ['kissat']], [DatabaseReader.FEATURES_GATE, ['kissat']],
          [DatabaseReader.FEATURES_BASE, DatabaseReader.FEATURES_GATE, ['kissat']],
          [DatabaseReader.FEATURES_BASE, ['glucose']], [DatabaseReader.FEATURES_GATE, ['glucose']],
          [DatabaseReader.FEATURES_BASE, DatabaseReader.FEATURES_GATE, ['glucose']],
          [DatabaseReader.FEATURES_BASE, ['cadical']],
          [DatabaseReader.FEATURES_GATE, ['cadical']],
          [DatabaseReader.FEATURES_BASE, DatabaseReader.FEATURES_GATE, ['cadical']],
          input_dbs]
output_merged = []
for combination in output:
    comb = []
    for elem in combination:
        comb = comb + elem
    output_merged.append(comb)

# plot_histograms_clustering('single_solver/single_solver_par2',
#                            0, ['selected_data'],
#                            [output_merged],
#                            ['base', 'gate', 'base gate', 'base kissat', 'gate kissat', 'base gate kissat',
#                             'base glucose', 'gate glucose', 'base gate glucose',
#                             'base cadical', 'gate cadical',
#                             'base gate cadical', 'base gate runtimes'],
#                            max_cluster_amount=20, columns=3,
#                            bin_step=10, height=0.11, output_file='single_solver/hist_single_solver_20', normalize=True)
#
# plot_histograms_clustering('single_solver/single_solver_par2',
#                            0, ['selected_data'],
#                            [output_merged],
#                            ['base', 'gate', 'base gate', 'base kissat', 'gate kissat', 'base gate kissat',
#                             'base glucose', 'gate glucose', 'base gate glucose',
#                             'base cadical', 'gate cadical',
#                             'base gate cadical', 'base gate runtimes'],
#                            max_cluster_amount=100, columns=3,
#                            bin_step=10, height=0.11, output_file='single_solver/hist_single_solver_100', normalize=True)
#
# plot_histograms_clustering('single_solver/single_solver_all_algos_par2',
#                            0, ['selected_data'],
#                            [output_merged],
#                            ['base', 'gate', 'base gate', 'base kissat', 'gate kissat', 'base gate kissat',
#                             'base glucose', 'gate glucose', 'base gate glucose',
#                             'base cadical', 'gate cadical',
#                             'base gate cadical', 'base gate runtimes'],
#                            max_cluster_amount=20, columns=3,
#                            bin_step=10, height=0.11, output_file='single_solver/hist_single_solver_all_algos_20',
#                            normalize=True)
#
# plot_histograms_clustering('single_solver/single_solver_all_algos_par2',
#                            0, ['selected_data'],
#                            [output_merged],
#                            ['base', 'gate', 'base gate', 'base kissat', 'gate kissat', 'base gate kissat',
#                             'base glucose', 'gate glucose', 'base gate glucose',
#                             'base cadical', 'gate cadical',
#                             'base gate cadical', 'base gate runtimes'],
#                            max_cluster_amount=100, columns=3,
#                            bin_step=10, height=0.11, output_file='single_solver/hist_single_solver_all_algos_100',
#                            normalize=True)

plot_boxplot_clustering('single_solver/single_solver_all_algos_par2',
                           0, ['selected_data'],
                           [output_merged],
                           ['base', 'gate', 'base gate', 'base kissat', 'gate kissat', 'base gate kissat',
                            'base glucose', 'gate glucose', 'base gate glucose',
                            'base cadical', 'gate cadical',
                            'base gate cadical', 'base gate runtimes'],
                           max_cluster_amount=20,  output_file='single_solver/box_single_solver_all_algos_20')