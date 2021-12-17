import itertools

from DataFormats.DbInstance import DbInstance
from run_plotting import plot_par2, plot_single_cluster_distribution_family
from util_scripts import DatabaseReader

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

db_instance = DbInstance()

plot_single_cluster_distribution_family('clustering_general/clustering_general_par2', db_instance, '',
                                 'best_clusters/clustering_best_cluster_base_gate_comb_100',
                                 'best_clusters/specific_clusters/1058', show_plot=True, use_mat_plot=True,
                                        use_dash_plot=True)
