import itertools

from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs

from yellowbrick.cluster import KElbowVisualizer

from DataAnalysis import feature_selection, scaling
from DataFormats.DbInstance import DbInstance
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

features = []
for feature_vector in input_dbs:
    features = features + feature_vector

db_instance = DbInstance(features)

comb_dict = {'selection_algorithm': 'NONE', 'scaling_algorithm': 'STANDARDSCALER'}

for comb in output_merged[1:]:
    print(comb)
    dataset_f, base_f, gate_f, solver_f, dataset, dataset_wh, base, base_wh, gate, gate_wh, solver, solver_wh = db_instance.generate_dataset(
        comb)
    feature_selected_data = feature_selection.feature_selection(dataset_wh, dataset_f, solver_wh, comb_dict)
    scaled_data = scaling.scaling(feature_selected_data, dataset_f, comb_dict)

    # Instantiate the clustering model and visualizer
    model = KMeans()
    visualizer = KElbowVisualizer(model, k=(1,100))

    visualizer.fit(scaled_data)        # Fit the data to the visualizer
    visualizer.show()        # Finalize and render the figure