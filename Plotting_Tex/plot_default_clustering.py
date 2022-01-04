import itertools
from run_plotting import plot_cbs_comparison

from util_scripts import DatabaseReader

dir = 'clustering_general'

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

plot_cbs_comparison([dir + '/clustering_general_par2'], 'vbs_sbs/vbs', 'vbs_sbs/sbs',
                      '',
                      0, ['selected_data'],
                      [output_merged[1:]],
                      ['base', 'gate', 'runtimes', 'base gate', 'base runtimes', 'gate runtimes', 'base gate runtimes'],
                      20, 200, output_file=dir + '/clustering_default_comb_all_dbs_new',
                      show_plot=False,
                      use_mat_plot=True, use_dash_plot=False, show_complete_legend=True)

plot_cbs_comparison([dir + '/clustering_general_par2'], 'vbs_sbs/vbs', 'vbs_sbs/sbs',
                      '',
                      0, ['cluster_algorithm', 'selected_data'],
                      [['KMEANS', 'AFFINITY', 'MEANSHIFT', 'SPECTRAL', 'AGGLOMERATIVE', 'OPTICS', 'GAUSSIAN', 'DBSCAN',
                        'BIRCH'], output_merged[1:]],
                      ['K-Means', 'Affintiy Propagation', 'Meanshift', 'Spectral Clustering', 'Agglomerative', 'OPTICS',
                       'Gaussian', 'DBSCAN', 'BIRCH'],
                      20, 200, output_file=dir + '/clustering_default_algo_all_dbs_new',
                      show_plot=False,
                      use_mat_plot=True, use_dash_plot=False, show_complete_legend=True)


# ----- Capped at 60s --------------------------------------------------------------------------------------------------

# plot_cpar2_comparison([dir + '/cap_60sec/general_clustering_capped_at_60_par2'], 'vbs_sbs/vbs', 'vbs_sbs/sbs',
#                       '',
#                       0, ['selected_data'],
#                       [output_merged[1:]],
#                       ['base', 'gate', 'runtimes', 'base gate', 'base runtimes', 'gate runtimes', 'base gate runtimes'],
#                       20, 200, output_file=dir + '/cap_60sec/clustering_default_capped_at_60_comb_all_dbs',
#                       show_plot=False,
#                       use_mat_plot=True, use_dash_plot=True)
#
# plot_cpar2_comparison([dir + '/cap_60sec/general_clustering_capped_at_60_par2'], 'vbs_sbs/vbs', 'vbs_sbs/sbs',
#                       '',
#                       0, ['cluster_algorithm', 'selected_data'],
#                       [['KMEANS', 'AFFINITY', 'MEANSHIFT', 'SPECTRAL', 'AGGLOMERATIVE', 'OPTICS', 'GAUSSIAN', 'DBSCAN',
#                         'BIRCH'], output_merged[1:]],
#                       ['K-Means', 'Affintiy Propagation', 'Meanshift', 'Spectral Clustering', 'Agglomerative', 'OPTICS',
#                        'Gaussian', 'DBSCAN', 'BIRCH'],
#                       20, 200, output_file=dir + '/cap_60sec/clustering_default_capped_at_60_algo_all_dbs',
#                       show_plot=False,
#                       use_mat_plot=True, use_dash_plot=True)

# Only base and gate ---------------------------------------------------------------------------------------------------

input_dbs = [DatabaseReader.FEATURES_BASE, DatabaseReader.FEATURES_GATE]
output = sum([list(map(list, itertools.combinations(input_dbs, i))) for i in range(len(input_dbs) + 1)], [])
output_merged = []
for combination in output:
    comb = []
    for elem in combination:
        comb = comb + elem
    output_merged.append(comb)


plot_cbs_comparison([dir + '/clustering_general_par2'], 'vbs_sbs/vbs', 'vbs_sbs/sbs',
                      '',
                    0, ['selected_data'],
                    [output_merged[1:]],
                    ['base', 'gate', 'base gate'],
                    20, 200, output_file=dir + '/clustering_default_comb_base_gate_new',
                    show_plot=False,
                    use_mat_plot=True, use_dash_plot=False, show_complete_legend=True)

plot_cbs_comparison([dir + '/clustering_general_par2'], 'vbs_sbs/vbs', 'vbs_sbs/sbs',
                      '',
                      0, ['cluster_algorithm', 'selected_data'],
                      [['KMEANS', 'AFFINITY', 'MEANSHIFT', 'SPECTRAL', 'AGGLOMERATIVE', 'OPTICS', 'GAUSSIAN', 'DBSCAN',
                        'BIRCH'], output_merged[1:]],
                      ['K-Means', 'Affintiy Propagation', 'Meanshift', 'Spectral Clustering', 'Agglomerative', 'OPTICS',
                       'Gaussian', 'DBSCAN', 'BIRCH'],
                      20, 200, output_file=dir + '/clustering_default_algo_base_gate_new',
                      show_plot=False,
                      use_mat_plot=True, use_dash_plot=False, show_complete_legend=True)

# ----- Capped at 60s --------------------------------------------------------------------------------------------------

# plot_cpar2_comparison([dir + '/cap_60sec/general_clustering_capped_at_60_par2'], 'vbs_sbs/vbs', 'vbs_sbs/sbs',
#                       '',
#                       0, ['selected_data'],
#                       [output_merged[1:]],
#                       ['base', 'gate', 'base gate'],
#                       20, 200, output_file=dir + '/cap_60sec/clustering_default_capped_at_60_comb_base_gate',
#                       show_plot=False,
#                       use_mat_plot=True, use_dash_plot=True)
#
# plot_cpar2_comparison([dir + '/cap_60sec/general_clustering_capped_at_60_par2'], 'vbs_sbs/vbs', 'vbs_sbs/sbs',
#                       '',
#                       0, ['cluster_algorithm', 'selected_data'],
#                       [['KMEANS', 'AFFINITY', 'MEANSHIFT', 'SPECTRAL', 'AGGLOMERATIVE', 'OPTICS', 'GAUSSIAN', 'DBSCAN',
#                         'BIRCH'], output_merged[1:]],
#                       ['K-Means', 'Affintiy Propagation', 'Meanshift', 'Spectral Clustering', 'Agglomerative', 'OPTICS',
#                        'Gaussian', 'DBSCAN', 'BIRCH'],
#                       20, 200, output_file=dir + '/cap_60sec/clustering_default_capped_at_60_algo_base_gate',
#                       show_plot=False,
#                       use_mat_plot=True, use_dash_plot=True)

