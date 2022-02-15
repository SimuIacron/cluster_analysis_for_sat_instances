import os
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.cluster import OPTICS
from DataAnalysis import feature_selection, scaling
from DataFormats.DbInstance import DbInstance
from UtilScripts.util import get_combinations_of_databases

output_merged, features = get_combinations_of_databases()

db_instance = DbInstance(features)

comb_dict = {
    'scaling_algorithm': 'SCALEMINUSPLUS1',
    'selection_algorithm': 'NONE',
    'scaling_technique': 'NORMALSCALE'
}

names = ['base', 'gate', 'runtimes', 'base_gate', 'base_runtimes', 'gate_runtimes', 'base_gate_runtimes']

for current_features, line, name in zip(output_merged, [0.5, 0.3, 0.7, 1.3, 0.9, 1.3, 1.8], names):
    print(current_features)
    dataset_f, base_f, gate_f, solver_f, dataset, dataset_wh, base, base_wh, gate, gate_wh, solver, solver_wh = db_instance.generate_dataset(
        current_features)
    feature_selected_data = feature_selection.feature_selection(dataset_wh, dataset_f, solver_wh, comb_dict)
    scaled_data = scaling.scaling(feature_selected_data, dataset_f, comb_dict)
    optics_model = OPTICS(min_samples=5)
    optics_model.fit(X=scaled_data)

    reachabilities = pd.Series(optics_model.reachability_).iloc[optics_model.ordering_]
    reachabilities.plot.bar(xticks=range(0, 2525, 50), figsize=(16, 8))
    # plt.axhline(y=line, color='red')

    plt.savefig(os.environ['TEXPATH'] + '/general_clustering_6/reachability/reachability_' + name + '.svg')
    plt.clf()
