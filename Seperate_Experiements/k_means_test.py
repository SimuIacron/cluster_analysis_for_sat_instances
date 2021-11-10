import pandas as pd
from sklearn.metrics import normalized_mutual_info_score, adjusted_mutual_info_score
import plotly.express as px
import plotly.graph_objects as go

from DataAnalysis import feature_reduction, scaling, clustering, scoring
from DataFormats.DbInstance import DbInstance
from DataFormats.InputData import InputDataCluster, InputDataScaling, InputDataFeatureSelection

from itertools import combinations

input = ['base', 'gate', 'solver']

output = sum([list(map(list, combinations(input, i))) for i in range(len(input) + 1)], [])

db_instance = DbInstance()

family_int = scoring.convert_families_to_int(db_instance.family_wh)
solver_int = scoring.get_best_solver_int(db_instance)



fig = go.Figure()
for comb in output[1:]:
    print(comb)

    db_instance.generate_dataset(comb)

    input_data_scaling = InputDataScaling(
        scaling_algorithm='SCALEMINUSPLUS1',
        scaling_technique='TIMESELECTBEST',
        scaling_k_best=3
    )

    input_data_feature_selection = InputDataFeatureSelection(
        selection_algorithm='MUTUALINFOK',
        seed=0,
        percentile_best=10)

    reduced_instance_list = feature_reduction.feature_reduction(
            db_instance.dataset_wh, db_instance.dataset_f, db_instance.solver_wh, input_data_feature_selection
        )

    instances_list_s = scaling.scaling(reduced_instance_list, db_instance.dataset_f, input_data_scaling)

    mutual_info_list = []
    k_range = range(1, 25)
    for k in k_range:

        input_data_cluster = InputDataCluster(
            cluster_algorithm='KMEANS',
            seed=0,
            n_clusters_k_means=k
        )

        # clustering
        (clusters, yhat) = clustering.cluster(instances_list_s, input_data_cluster)

        mutual_info = adjusted_mutual_info_score(solver_int, yhat)
        mutual_info_list.append(mutual_info)

    fig.add_trace(go.Scatter(x=list(k_range), y=mutual_info_list, mode='lines', name=" ".join(str(x) for x in comb)))

fig.show()






