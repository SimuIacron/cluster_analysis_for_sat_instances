import itertools

from DataFormats.DbInstance import DbInstance
from run_plotting_histograms import plot_boxplot_family
from util_scripts import DatabaseReader

output_directory = '/family'
dpi = 192

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

features = []
for feature_vector in input_dbs:
    features = features + feature_vector
db_instance = DbInstance(features)

plot_boxplot_family(db_instance, output_file=output_directory + '/family_distribution', dpi=192)
