import itertools
from run_plotting import plot_cbs_comparison
from run_plotting_histograms import plot_boxplot_clustering

from util_scripts import DatabaseReader

dir = 'single_feature_clustering'
dpi = 96
y_axis_range = [3500, 3960]

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

features = [[item] for item in DatabaseReader.FEATURES_BASE]
plot_cbs_comparison([dir + '/single_feature_clustering_base_par2'], 'vbs_sbs/vbs', 'vbs_sbs/sbs',
                    '',  # 'Best CPar2 scores for clusterings with single base features',
                    0, ['selected_data'],
                    [features],
                    DatabaseReader.FEATURES_BASE, 20, 100,
                    output_file='/preprocessing/single_feature_clustering/single_feature_clustering_base_plot', show_plot=False,
                    use_mat_plot=True, use_dash_plot=True, show_complete_legend=False)

plot_boxplot_clustering(dir + '/single_feature_clustering_base_par2',
                        0, ['selected_data'],
                        [features],
                        DatabaseReader.FEATURES_BASE, max_cluster_amount=20, angle=90, dpi=dpi,
                        y_axis_range=y_axis_range,
                        output_file='/preprocessing/single_feature_clustering/box_single_feature_clustering_base', show_plot=False)

# ----------------------------------------------------------------------------------------------------------------------

features = [[item] for item in DatabaseReader.FEATURES_GATE]
plot_cbs_comparison([dir + '/single_feature_clustering_gate_par2'], 'vbs_sbs/vbs', 'vbs_sbs/sbs',
                    '',  # 'Best CPar2 scores for clusterings with single gate features',
                    0, ['selected_data'],
                    [features],
                    DatabaseReader.FEATURES_GATE, 20, 100,
                    output_file='/preprocessing/single_feature_clustering/single_feature_clustering_gate_plot', show_plot=False,
                    use_mat_plot=True, use_dash_plot=True, show_complete_legend=False)

plot_boxplot_clustering(dir + '/single_feature_clustering_gate_par2',
                        0, ['selected_data'],
                        [features],
                        DatabaseReader.FEATURES_GATE, max_cluster_amount=20, angle=90, dpi=dpi,
                        y_axis_range=y_axis_range,
                        output_file='/preprocessing/single_feature_clustering/box_single_feature_clustering_gate', show_plot=False)

# ----------------------------------------------------------------------------------------------------------------------

features = [[item] for item in temp_solver_features]
plot_cbs_comparison([dir + '/single_feature_clustering_runtimes_par2'], 'vbs_sbs/vbs', 'vbs_sbs/sbs',
                    '',  # 'Best CPar2 scores for clusterings with single runtimes features',
                    0, ['selected_data'],
                    [features],
                    DatabaseReader.FEATURES_SOLVER, 20, 100,
                    output_file='/preprocessing/single_feature_clustering/single_feature_clustering_runtimes_plot', show_plot=False,
                    use_mat_plot=True, use_dash_plot=True, show_complete_legend=False)

plot_boxplot_clustering(dir + '/single_feature_clustering_runtimes_par2',
                        0, ['selected_data'],
                        [features],
                        temp_solver_features, max_cluster_amount=20, angle=90, dpi=dpi,
                        y_axis_range=y_axis_range,
                        output_file='/preprocessing/single_feature_clustering/box_single_feature_clustering_runtimes',
                        show_plot=False)
