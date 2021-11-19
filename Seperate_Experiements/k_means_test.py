from collections import Counter

import numpy as np
import pandas as pd
from numpy import mean
from sklearn.metrics import normalized_mutual_info_score, adjusted_mutual_info_score, completeness_score
import plotly.express as px
import plotly.graph_objects as go

import exportFigure
from DataAnalysis import feature_reduction, scaling, clustering, scoring
from DataFormats.DbInstance import DbInstance
from DataFormats.InputData import InputDataCluster, InputDataScaling, InputDataFeatureSelection

from itertools import combinations

input_dbs = ['base', 'gate', 'solver']

output = sum([list(map(list, combinations(input_dbs, i))) for i in range(len(input_dbs) + 1)], [])

db_instance = DbInstance()

family_int = scoring.convert_families_to_int(db_instance.family_wh)
solver_int = scoring.get_best_solver_int(db_instance)
unsat_sat_int = scoring.convert_sat_unsat_to_int(db_instance.result_wh)

plot_name_1 = "012_kmeans_par2_10_best"

fig_1 = go.Figure(layout=go.Layout(
        title=go.layout.Title(text=plot_name_1)
    ))
fig_2 = go.Figure(layout=go.Layout(
        title=go.layout.Title(text="009_kmeans_par2_scale01_25")
    ))
fig_3 = go.Figure(layout=go.Layout(
        title=go.layout.Title(text=plot_name_1)
    ))
legendCounter = -1
for comb in output[1:]:
    legendCounter = legendCounter + 1
    print(comb)

    db_instance.generate_dataset(comb)

    input_data_scaling = InputDataScaling(
        scaling_algorithm='SCALEMINUSPLUS1',
        scaling_technique='TIMESELECTBEST',
        scaling_k_best=10
    )

    input_data_feature_selection = InputDataFeatureSelection(
        selection_algorithm='NONE',
        seed=0,
        percentile_best=30)

    reduced_instance_list = feature_reduction.feature_reduction(
            db_instance.dataset_wh, db_instance.dataset_f, db_instance.solver_wh, input_data_feature_selection
        )

    instances_list_s = scaling.scaling(reduced_instance_list, db_instance.dataset_f, input_data_scaling)

    solver_list = []
    family_list = []
    list_1 = []
    keys_1 = []
    values_1 = []
    text_1 = []
    sizes_1 = []
    # k_range = [i * 5 for i in range(1, 100)]
    k_range = range(1, 25)
    # eps_range = np.arange(0.05, 2, 0.05)
    for k in k_range:

        input_data_cluster = InputDataCluster(
            cluster_algorithm='KMEANS',
            seed=0,
            n_clusters_k_means=k
        )

        # clustering
        (clusters, yhat) = clustering.cluster(instances_list_s, input_data_cluster)

        # value = normalized_mutual_info_score(unsat_sat_int, yhat)
        # solver_list.append(value)
        # value = adjusted_mutual_info_score(family_int, yhat)
        # family_list.append(value)
        # value = completeness_score(solver_int, yhat)
        # solver_list.append(value)
        # value = completeness_score(family_int, yhat)
        # family_list.append(value)
        # value = scoring.van_dongen_normalized(solver_int, yhat)
        # solver_list.append(value)
        # value = scoring.van_dongen_normalized(family_int, yhat)
        # family_list.append(value)
        # value = min([scoring.score_solvers_on_linear_rank_cluster(yhat, i, db_instance, 5000)[1] for i in clusters])
        # solver_list.append(value)
        # value = max([scoring.score_solvers_on_linear_rank_cluster(yhat, i, db_instance, 5000)[1] for i in clusters])
        # family_list.append(value)
        # value = mean([scoring.score_solvers_on_linear_rank_cluster(yhat, i, db_instance, 5000)[1] for i in clusters])
        # list_1.append(value)
        value, scores_clusters, cluster_algo = scoring.score_clustering_par2(clusters, yhat, db_instance, 5000)
        list_1.append(value)
        keys_1 = keys_1 + ([k] * len(scores_clusters))
        values_1 = values_1 + scores_clusters
        cluster_amount = Counter(yhat)
        sizes_1 = sizes_1 + [cluster_amount[idx] for idx in range(len(cluster_amount))]
        text_1 = text_1 + [(str(cluster_amount[idx]) + ' ' + str(cluster_algo[idx])) for idx in range(len(cluster_amount))]

    # fig_1.add_trace(go.Scatter(x=list(k_range), y=solver_list, mode='lines', name=" ".join(str(x) for x in comb)))
    # fig_2.add_trace(go.Scatter(x=list(k_range), y=family_list, mode='lines', name=" ".join(str(x) for x in comb)))
    fig_3.add_trace(go.Scatter(x=list(k_range), y=list_1, mode='lines', name=" ".join(str(x) for x in comb), legendgroup=legendCounter))
    fig_3.add_trace(go.Scatter(x=keys_1, y=values_1, mode='markers', marker=dict(size=[np.sqrt(item/np.pi)*2 for item in sizes_1]), text=text_1, name=" ".join(str(x) for x in comb), legendgroup=legendCounter, showlegend=False))

# exportFigure.export_plot_as_html(fig_1, plot_name_1)
# exportFigure.export_plot_as_html(fig_2, '007_kmeans_worst_scale_solver')
exportFigure.export_plot_as_html(fig_3, plot_name_1)

# fig_1.show()
# fig_2.show()
fig_3.show()






