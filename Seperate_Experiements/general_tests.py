from itertools import combinations

import exportFigure
from DataAnalysis import scoring, feature_reduction, scaling
from DataFormats.DbInstance import DbInstance
from sklearn.metrics import normalized_mutual_info_score, adjusted_mutual_info_score

from DataFormats.InputData import InputDataScaling, InputDataFeatureSelection

db_instance = DbInstance()

family_int = scoring.convert_families_to_int(db_instance.family_wh)
solver_int = scoring.get_best_solver_int(db_instance)

print(adjusted_mutual_info_score(family_int, solver_int))

input_dbs = ['base', 'gate', 'solver']
output = sum([list(map(list, combinations(input_dbs, i))) for i in range(len(input_dbs) + 1)], [])

for comb in output[1:]:
    print(comb)

    db_instance.generate_dataset(comb)
    print("Original feature amount: " + str(len(db_instance.dataset_f)))

    input_data_scaling = InputDataScaling(
        scaling_algorithm='STANDARDSCALER',
        scaling_technique='TIMESELECTBEST',
        scaling_k_best=3
    )

    input_data_feature_selection = InputDataFeatureSelection(
        selection_algorithm='MUTUALINFO',
        seed=0,
        percentile_best=30)

    instances_list_s = scaling.scaling(db_instance.dataset_wh, db_instance.dataset_f, input_data_scaling)
    reduced_instance_list = feature_reduction.feature_reduction(
            instances_list_s, db_instance.dataset_f, db_instance.solver_wh, input_data_feature_selection
        )


