import pandas as pd
from sklearn.metrics import normalized_mutual_info_score, adjusted_mutual_info_score
import plotly.express as px
import plotly.graph_objects as go

import exportFigure
from DataAnalysis import feature_reduction, scaling, clustering, scoring
from DataFormats.DbInstance import DbInstance
from DataFormats.InputData import InputDataCluster, InputDataScaling, InputDataFeatureSelection

from itertools import combinations

input = ['base', 'gate', 'solver']

output = sum([list(map(list, combinations(input, i))) for i in range(len(input) + 1)], [])

db_instance = DbInstance()

family_int = scoring.convert_families_to_int(db_instance.family_wh)
solver_int = scoring.get_best_solver_int(db_instance)



fig_family = go.Figure(layout=go.Layout(
        title=go.layout.Title(text="Family")
    ))
fig_solver = go.Figure(layout=go.Layout(
        title=go.layout.Title(text="Solver")
    ))
for comb in output[1:]:
    print(comb)

    db_instance.generate_dataset(comb)

    input_data_scaling = InputDataScaling(
        scaling_algorithm='SCALEMINUSPLUS1',
        scaling_technique='TIMESELECTBEST',
        scaling_k_best=3
    )

    input_data_feature_selection = InputDataFeatureSelection(
        selection_algorithm='NONE',
        seed=0,
        percentile_best=10)

    reduced_instance_list = feature_reduction.feature_reduction(
            db_instance.dataset_wh, db_instance.dataset_f, db_instance.solver_wh, input_data_feature_selection
        )

    instances_list_s = scaling.scaling(reduced_instance_list, db_instance.dataset_f, input_data_scaling)

    mutual_info_solver_list = []
    mutual_info_family_list = []
    k_range = [i * 5 for i in range(1, 100)]
    for k in k_range:

        input_data_cluster = InputDataCluster(
            cluster_algorithm='KMEANS',
            seed=0,
            n_clusters_k_means=k
        )

        # clustering
        (clusters, yhat) = clustering.cluster(instances_list_s, input_data_cluster)

        mutual_info = normalized_mutual_info_score(solver_int, yhat)
        mutual_info_solver_list.append(mutual_info)
        mutual_info = normalized_mutual_info_score(family_int, yhat)
        mutual_info_family_list.append(mutual_info)

    fig_solver.add_trace(go.Scatter(x=list(k_range), y=mutual_info_solver_list, mode='lines', name=" ".join(str(x) for x in comb)))
    fig_family.add_trace(go.Scatter(x=list(k_range), y=mutual_info_family_list, mode='lines', name=" ".join(str(x) for x in comb)))

exportFigure.export_plot_as_html(fig_family, '001_kmeans_mut_family_normalized')
exportFigure.export_plot_as_html(fig_solver, '001_kmeans_mut_solver_normalized')

fig_family.show()
fig_solver.show()






