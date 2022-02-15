from sklearn.metrics import normalized_mutual_info_score
from PlottingAndEvaluationFunctions.func_plot_changes_between_clusterings import filter_clustering_settings, \
    plot_homogeneity_change
from run_experiments import read_json
from util_scripts.util import get_combinations_of_databases

version = 6
input_file_clustering = 'clustering_general_v{ver}/standardscaler/general_clustering_{ver}_par2'.format(ver=version)
output_file = '/general_clustering_{ver}/clusterings_mutual_info_change/'.format(ver=version)
names = ['base', 'gate', 'runtimes', 'base_gate', 'base_runtimes', 'gate_runtimes', 'base_gate_runtimes']

# enable/disable these to plot the according algorithm
plot_kmeans = True
plot_dbscan = False

clusterings_read = read_json(input_file_clustering)

combinations, features = get_combinations_of_databases()
for i, combination in enumerate(combinations):
    print(combination)

    if plot_kmeans:
        filtered_clustering = filter_clustering_settings(clusterings_read,
                                                         ['cluster_algorithm', 'selected_data'],
                                                         [['KMEANS'], [combination]])

        plot_homogeneity_change(filtered_clustering, 'n_clusters_k_means', 'k', normalized_mutual_info_score,
                                'Normalized Mutual Information',
                                output_file=output_file + 'kmeans/mutual_info_change_' + names[i])

    if plot_dbscan:
        filtered_clustering = filter_clustering_settings(clusterings_read,
                                                         ['cluster_algorithm', 'selected_data', 'min_samples_dbscan'],
                                                         [['DBSCAN'], [combination], [5]])

        plot_homogeneity_change(filtered_clustering, 'eps_dbscan', 'Epsilon', normalized_mutual_info_score,
                                'Normalized Mutual Information',
                                output_file=output_file + 'dbscan/mutual_info_change_minsamples5_' + names[i])