import itertools

from run_plotting import plot_best_cluster_comparison
from util_scripts import DatabaseReader

output_directory = '/clustering_general_v2'
input_file = 'clustering_general_v2/general_clustering_2_par2'

input_dbs = [DatabaseReader.FEATURES_BASE, DatabaseReader.FEATURES_GATE]
output = sum([list(map(list, itertools.combinations(input_dbs, i))) for i in range(len(input_dbs) + 1)], [])
output_merged = []
for combination in output:
    comb = []
    for elem in combination:
        comb = comb + elem
    output_merged.append(comb)

plot_best_cluster_comparison([input_file], '',
                          0, ['cluster_algorithm', 'selected_data'],
                          [['KMEANS', 'AFFINITY', 'MEANSHIFT', 'SPECTRAL', 'AGGLOMERATIVE', 'OPTICS', 'GAUSSIAN',
                          'DBSCAN',
                          'BIRCH'], output_merged[1:]],
                          ['K-Means', 'Affintiy Propagation', 'Meanshift', 'Spectral Clustering', 'Agglomerative',
                          'OPTICS',
                          'Gaussian', 'DBSCAN', 'BIRCH'],
                          max_cluster_amount=10000, min_cluster_size=10, cutoff=100000,
                          output_file_json=output_directory + '/single_clusters/sorted_best_clusters_base_gate_10_10000',
                          show_plot=False, use_mat_plot=False, use_dash_plot=False)