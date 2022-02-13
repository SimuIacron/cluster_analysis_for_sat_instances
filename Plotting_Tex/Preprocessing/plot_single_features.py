import itertools
from run_plotting import plot_cbs_comparison
from run_plotting_histograms import plot_boxplot_clustering

from util_scripts import DatabaseReader
from util_scripts.util import get_combinations_of_databases

version = 6
input_file_single_features = 'clustering_general_v{ver}/standardscaler/single_features/general_clustering_{ver}_single_features_standardscaler_par2'.format(ver=version)
input_file_sbs = 'clustering_general_v{ver}/sbs_{ver}'.format(ver=version)
output_file = '/general_clustering_{ver}/single_features/'.format(ver=version)


dpi = 192
angle=90
y_axis_range = [3550, 3960]
remove_time = 25
max_id=24856

temp_solver_features = DatabaseReader.FEATURES_SOLVER.copy()
temp_solver_features.pop(14)
temp_solver_features.pop(7)

output_merged = get_combinations_of_databases()

features = [[item] for item in DatabaseReader.FEATURES_BASE]
# plot_cbs_comparison([dir + '/single_feature_clustering_base_par2'], 'vbs_sbs/vbs', 'vbs_sbs/sbs',
#                     '',  # 'Best CPar2 scores for clusterings with single base features',
#                     0, ['selected_data'],
#                     [features],
#                     DatabaseReader.FEATURES_BASE, 20, 100,
#                     output_file='/preprocessing/single_feature_clustering/single_feature_clustering_base_plot', show_plot=False,
#                     use_mat_plot=True, use_dash_plot=True, show_complete_legend=False)

plot_boxplot_clustering(input_file_single_features,
                        0, ['selected_data'],
                        [features],
                        DatabaseReader.FEATURES_BASE, max_cluster_amount=20, angle=angle, dpi=dpi,
                        y_axis_range=y_axis_range,
                        output_file=output_file + 'box_single_feature_clustering_base', show_plot=False,
                        remove_box_if_all_values_in_range_of_sbs=remove_time, sbs_file=input_file_sbs, max_id=max_id)

# ----------------------------------------------------------------------------------------------------------------------

features = [[item] for item in DatabaseReader.FEATURES_GATE]
# plot_cbs_comparison([dir + '/single_feature_clustering_gate_par2'], 'vbs_sbs/vbs', 'vbs_sbs/sbs',
#                     '',  # 'Best CPar2 scores for clusterings with single gate features',
#                     0, ['selected_data'],
#                     [features],
#                     DatabaseReader.FEATURES_GATE, 20, 100,
#                     output_file='/preprocessing/single_feature_clustering/single_feature_clustering_gate_plot', show_plot=False,
#                     use_mat_plot=True, use_dash_plot=True, show_complete_legend=False)

plot_boxplot_clustering(input_file_single_features,
                        0, ['selected_data'],
                        [features],
                        DatabaseReader.FEATURES_GATE, max_cluster_amount=20, angle=angle, dpi=dpi,
                        y_axis_range=y_axis_range,
                        output_file=output_file + 'box_single_feature_clustering_gate', show_plot=False,
                        remove_box_if_all_values_in_range_of_sbs=remove_time, sbs_file=input_file_sbs, max_id=max_id)

# ----------------------------------------------------------------------------------------------------------------------

features = [[item] for item in temp_solver_features]
# plot_cbs_comparison([dir + '/single_feature_clustering_runtimes_par2'], 'vbs_sbs/vbs', 'vbs_sbs/sbs',
#                     '',  # 'Best CPar2 scores for clusterings with single runtimes features',
#                     0, ['selected_data'],
#                     [features],
#                     DatabaseReader.FEATURES_SOLVER, 20, 100,
#                     output_file='/preprocessing/single_feature_clustering/single_feature_clustering_runtimes_plot', show_plot=False,
#                     use_mat_plot=True, use_dash_plot=True, show_complete_legend=False)

plot_boxplot_clustering(input_file_single_features,
                        0, ['selected_data'],
                        [features],
                        temp_solver_features, max_cluster_amount=20, angle=angle, dpi=dpi,
                        y_axis_range=y_axis_range,
                        output_file=output_file + 'box_single_feature_clustering_runtimes',
                        show_plot=False,
                        remove_box_if_all_values_in_range_of_sbs=remove_time, sbs_file=input_file_sbs, max_id=max_id)
