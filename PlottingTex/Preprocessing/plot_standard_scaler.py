import itertools
from PlottingAndEvaluationFunctions.func_plot_clustering_algorithm_comparison import plot_histograms_clustering, plot_boxplot_clustering

from DataFormats import DatabaseReader

dpi = 192
angle = 20
max_cluster_amount = 35
max_id = 24856

names = ['Linear Scaling', 'Standard Scaling']

version = 6
input_file_clustering_standard_scaler = 'clustering_general_v{ver}/general_clustering_{ver}_par2'.format(ver=version)
input_file_clustering_linear_scaler = \
    'clustering_general_v{ver}/linearscaler/general_clustering_{ver}_linearscaler_par2'.format(ver=version)
input_file_sbs = 'clustering_general_v{ver}/sbs_{ver}'.format(ver=version)
output_file = '/general_clustering_{ver}/preprocessing/scaling/'.format(ver=version)
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

plot_histograms_clustering([input_file_clustering_linear_scaler, input_file_clustering_standard_scaler], input_file_sbs,
                           0, ['scaling_algorithm'],
                           [['SCALEMINUSPLUS1', 'STANDARDSCALER']],
                           names,
                           max_cluster_amount=max_cluster_amount, columns=2,
                           bin_step=10, height=5000,
                           output_file=output_file + 'hist_standardscaler_linearscaler_all',
                           dpi=dpi, max_id=max_id)

plot_boxplot_clustering([input_file_clustering_linear_scaler, input_file_clustering_standard_scaler],
                        0, ['scaling_algorithm'],
                        [['SCALEMINUSPLUS1', 'STANDARDSCALER']],
                        names,
                        max_cluster_amount=max_cluster_amount, angle=angle,
                        output_file=output_file + 'box_standardscaler_linearscaler_all',
                        dpi=dpi, max_id=max_id)

input_dbs = [DatabaseReader.FEATURES_BASE, DatabaseReader.FEATURES_GATE]
output = sum([list(map(list, itertools.combinations(input_dbs, i))) for i in range(len(input_dbs) + 1)], [])
output_merged = []
for combination in output:
    comb = []
    for elem in combination:
        comb = comb + elem
    output_merged.append(comb)

plot_histograms_clustering([input_file_clustering_linear_scaler, input_file_clustering_standard_scaler], input_file_sbs,
                           0, ['scaling_algorithm', 'selected_data'],
                           [['SCALEMINUSPLUS1', 'STANDARDSCALER'], output_merged[1:]],
                           names,
                           max_cluster_amount=max_cluster_amount, columns=2,
                           bin_step=10, height=3000,
                           output_file=output_file + 'hist_standardscaler_linearscaler_base_gate',
                           dpi=dpi, max_id=max_id)

plot_boxplot_clustering([input_file_clustering_linear_scaler, input_file_clustering_standard_scaler],
                        0, ['scaling_algorithm', 'selected_data'],
                        [['SCALEMINUSPLUS1', 'STANDARDSCALER'], output_merged[1:]],
                        names,
                        max_cluster_amount=max_cluster_amount, angle=angle,
                        output_file=output_file + 'box_standardscaler_linearscaler_base_gate',
                        dpi=dpi, max_id=max_id)


# CHANGE ax.set_position to multiply by 0.7 instead of 0.8!
# plot_cbs_comparison(['scaling_standardscaler/standardscaler_linearscaler_clustering_par2'], 'vbs_sbs/vbs', 'vbs_sbs/sbs',
#                      '',
#                      0, ['scaling_algorithm'],
#                       [['SCALEMINUSPLUS1', 'STANDARDSCALER']], ['Linear Scaling to [-1,+1]', 'Standard Scaling'], 20, 100,
#                       output_file='scaling_standardscaler/standardscaler_linearscaler_clustering_par2', show_plot=False)
