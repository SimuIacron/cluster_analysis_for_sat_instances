import util_scripts.util
from PlottingAndEvaluationFunctions.func_plot_single_clusters import boxplot_cluster_feature_distribution
from PlottingAndEvaluationFunctions.func_stochastic_values_dataset import calculate_stochastic_value_for_dataset
from DataFormats.DbInstance import DbInstance
from util_scripts.util import get_combinations_of_databases

version = '6'
input_file_cluster = 'clustering_general_v{ver}/standardscaler/general_clustering_{ver}_clusters'.format(ver=version)
input_file_clustering = 'clustering_general_v{ver}/standardscaler/general_clustering_{ver}'.format(ver=version)
input_file_dataset = 'clustering_general_v{ver}/stochastic_values_dataset'.format(ver=version)
input_file_sbs = 'clustering_general_v{ver}/sbs_{ver}'.format(ver=version)
output_file = '/general_clustering_{ver}/'.format(ver=version)
dpi = 120

excluded = ['levels_min'
                 'levels_none_mean', 'levels_none_variance', 'levels_none_min', 'levels_none_max',
                 'levels_none_entropy',
                 'horn_vars_min',
                 'inv_horn_vars_min',
                 'balance_vars_mean', 'balance_vars_variance', 'balance_vars_min', 'balance_vars_max',
                 'balance_vars_entropy', 'vcg_vdegrees_min',
                 'vg_degrees_min'
                 ]

output_merged, features = get_combinations_of_databases()

db_instance = DbInstance(features)

values = calculate_stochastic_value_for_dataset(db_instance)
boxplot_cluster_feature_distribution(values, db_instance, dpi=dpi, use_base=True, use_gate=False,
                                     angle=90, output_file=output_file + 'dataset_base',
                                     show_plot=False, exclude_features=excluded)
boxplot_cluster_feature_distribution(values, db_instance, dpi=dpi, use_base=False, use_gate=True,
                                     angle=90, output_file=output_file + 'dataset_gate',
                                     show_plot=False, exclude_features=excluded)

base = util_scripts.util.rotateNestedLists(values['base_01'])
gate = util_scripts.util.rotateNestedLists(values['gate_01'])

empty_features_base = []
for i, feature_base in enumerate(base):
    has_different_values = False
    for item in feature_base:
        if item != 0:
            has_different_values = True
            break

    if not has_different_values:
        empty_features_base.append(db_instance.base_f[i])

empty_features_gate = []
for i, feature_gate in enumerate(gate):
    has_different_values = False
    for item in feature_gate:
        if item != 0:
            has_different_values = True
            break

    if not has_different_values:
        empty_features_gate.append(db_instance.gate_f[i])

pass



