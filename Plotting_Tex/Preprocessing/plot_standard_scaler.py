import itertools
from run_plotting import plot_cbs_comparison
from run_plotting_histograms import plot_histograms_clustering, plot_boxplot_clustering

from util_scripts import DatabaseReader

dir = 'scaling_standardscaler'
sbs_file = 'vbs_sbs/sbs'
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

plot_histograms_clustering(dir + '/standardscaler_linearscaler_clustering_par2', sbs_file,
                           0, ['scaling_algorithm'],
                           [['SCALEMINUSPLUS1', 'STANDARDSCALER']],
                           ['[-1,+1]', 'Standard Scaler'],
                           max_cluster_amount=20, columns=2,
                           bin_step=10, height=550,
                           output_file='/preprocessing/standardscaler_linearscaler/hist_standardscaler_linearscaler_all')

plot_boxplot_clustering(dir + '/standardscaler_linearscaler_clustering_par2',
                           0, ['scaling_algorithm'],
                           [['SCALEMINUSPLUS1', 'STANDARDSCALER']],
                           ['[-1,+1]', 'Standard Scaler'],
                           max_cluster_amount=20, angle=0,
                           output_file='/preprocessing/standardscaler_linearscaler/box_standardscaler_linearscaler_all')

input_dbs = [DatabaseReader.FEATURES_BASE, DatabaseReader.FEATURES_GATE]
output = sum([list(map(list, itertools.combinations(input_dbs, i))) for i in range(len(input_dbs) + 1)], [])
output_merged = []
for combination in output:
    comb = []
    for elem in combination:
        comb = comb + elem
    output_merged.append(comb)

plot_histograms_clustering(dir + '/standardscaler_linearscaler_clustering_par2', sbs_file,
                           0, ['scaling_algorithm', 'selected_data'],
                           [['SCALEMINUSPLUS1', 'STANDARDSCALER'], output_merged[1:]],
                           ['[-1,+1]', 'Standard Scaler'],
                           max_cluster_amount=20, columns=2,
                           bin_step=10, height=500,
                           output_file='/preprocessing/standardscaler_linearscaler/hist_standardscaler_linearscaler_base_gate')

plot_boxplot_clustering(dir + '/standardscaler_linearscaler_clustering_par2',
                           0, ['scaling_algorithm', 'selected_data'],
                           [['SCALEMINUSPLUS1', 'STANDARDSCALER'], output_merged[1:]],
                           ['[-1,+1]', 'Standard Scaler'],
                           max_cluster_amount=20, angle=0,
                           output_file='/preprocessing/standardscaler_linearscaler/box_standardscaler_linearscaler_base_gate')



# CHANGE ax.set_position to multiply by 0.7 instead of 0.8!
# plot_cbs_comparison(['scaling_standardscaler/standardscaler_linearscaler_clustering_par2'], 'vbs_sbs/vbs', 'vbs_sbs/sbs',
#                      '',
#                      0, ['scaling_algorithm'],
#                       [['SCALEMINUSPLUS1', 'STANDARDSCALER']], ['Linear Scaling to [-1,+1]', 'Standard Scaling'], 20, 100,
#                       output_file='scaling_standardscaler/standardscaler_linearscaler_clustering_par2', show_plot=False)
