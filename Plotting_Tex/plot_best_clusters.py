import itertools
from run_plotting import plot_best_cluster_comparison

from util_scripts import DatabaseReader

dir = 'best_clusters'

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

plot_best_cluster_comparison(['clustering_general/clustering_general_par2'],
                      '',  # 'Par2 scores of the best clusters in clusterings using combinations of base, gate, runtimes',
                      0, ['selected_data'],
                      [output_merged[1:]],
                      ['base', 'gate', 'runtimes', 'base gate', 'base runtimes', 'gate runtimes', 'base gate runtimes'],
                      100, 100, 5000, output_file=dir + '/clustering_best_cluster_all_5000',
                      show_plot=False,
                      use_mat_plot=True, use_dash_plot=True)