import itertools
from run_plotting import plot_cpar2_comparison

from util_scripts import DatabaseReader


dir = 'single_feature_clustering'

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
plot_cpar2_comparison([dir + '/single_feature_clustering_base_par2'], 'vbs_sbs/vbs', 'vbs_sbs/sbs',
                      '',
                      0, ['selected_data'],
                      [features],
                      DatabaseReader.FEATURES_BASE,
                      20, 200, output_file=dir + '/single_feature_clustering_base_plot',
                      show_plot=False,
                      use_mat_plot=True, use_dash_plot=True)
