from numpy import mean
from sklearn.metrics import adjusted_mutual_info_score
import plotly.graph_objects as go

import DatabaseReader
import exportFigure
from DataAnalysis import feature_selection, scaling, clustering
from DataAnalysis.Evaluation import scoring_util, scoring
from DataFormats.DbInstance import DbInstance
from DataFormats.InputData import InputDataCluster, InputDataScaling, InputDataFeatureSelection

from itertools import combinations

input = ['base', 'gate', 'solver']

output = sum([list(map(list, combinations(input, i))) for i in range(len(input) + 1)], [])

db_instance = DbInstance()

family_int = scoring_util.convert_families_to_int(db_instance.family_wh)
solver_int = scoring_util.convert_best_solver_int(db_instance)

surface_1 = []
surface_2 = []
surface_3 = []
surface_4 = []


for comb in output[1:]:
    print(comb)

    db_instance.generate_dataset(comb)


    list_1 = []
    list_2 = []
    list_3 = []
    list_4 = []

    k_range = range(5, 50, 5)
    percentile_range = range(5, 50, 5)
    for k in k_range:
        current_list_1 = []
        current_list_2 = []
        current_list_3 = []
        current_list_4 = []
        for percentile in percentile_range:

            input_data_feature_selection = InputDataFeatureSelection(
                selection_algorithm='MUTUALINFO',
                seed=0,
                percentile_best=percentile)

            reduced_instance_list = feature_selection.feature_selection(
                db_instance.dataset_wh, db_instance.dataset_f, db_instance.solver_wh, input_data_feature_selection
            )

            input_data_scaling = InputDataScaling(
                scaling_algorithm='SCALEMINUSPLUS1',
                scaling_technique='TIMESELECTBEST',
                scaling_k_best=3
            )

            instances_list_s = scaling.scaling(reduced_instance_list, db_instance.dataset_f, input_data_scaling)

            input_data_cluster = InputDataCluster(
                cluster_algorithm='KMEANS',
                seed=0,
                n_clusters_k_means=k
            )

            # clustering
            (clusters, yhat) = clustering.cluster(instances_list_s, input_data_cluster)

            value = adjusted_mutual_info_score(solver_int, yhat)
            current_list_1.append(value)
            value = adjusted_mutual_info_score(family_int, yhat)
            current_list_2.append(value)
            value = mean([scoring.score_solvers_on_linear_rank_cluster(yhat, i, db_instance, DatabaseReader.TIMEOUT)[1]
                          for i in clusters])
            current_list_3.append(value)
            value = mean(
                [scoring.score_solvers_on_rank_cluster(yhat, i, db_instance, 
                                                       [1000, 2000, 3000, 4000, DatabaseReader.TIMEOUT],
                                                       [5, 4, 3, 2, 1])[1] for i in clusters])
            current_list_4.append(value)

        list_1.append(current_list_1)
        list_2.append(current_list_2)
        list_3.append(current_list_3)
        list_4.append(current_list_4)

    surface_1.append(go.Surface(z=list_1, x=list(k_range), y=list(percentile_range), opacity=0.7, name=" ".join(str(x) for x in comb)))
    surface_2.append(go.Surface(z=list_2, x=list(k_range), y=list(percentile_range), opacity=0.7, name=" ".join(str(x) for x in comb)))
    surface_3.append(go.Surface(z=list_3, x=list(k_range), y=list(percentile_range), opacity=0.7,
                                     name=" ".join(str(x) for x in comb)))
    surface_3.append(go.Surface(z=list_4, x=list(k_range), y=list(percentile_range), opacity=0.7,
                                     name=" ".join(str(x) for x in comb)))


fig_1 = go.Figure(layout=go.Layout(
        title=go.layout.Title(text="Mutual info family")
    ),
    data=surface_1
)

fig_2 = go.Figure(layout=go.Layout(
        title=go.layout.Title(text="Mutual info solver")
    ),
    data=surface_2
)

fig_3 = go.Figure(layout=go.Layout(
        title=go.layout.Title(text="Mean Score solver linear rank")
    ),
    data=surface_3
)

fig_4 = go.Figure(layout=go.Layout(
        title=go.layout.Title(text="Mean score solver rank")
    ),
    data=surface_4
)

exportFigure.export_plot_as_html(fig_1, '008_kmeans_mut_family_selection_mut')
exportFigure.export_plot_as_html(fig_2, '008_kmeans_mut_solver_selection_mut')
exportFigure.export_plot_as_html(fig_3, '008_kmeans_linear_rank_selection_mut')
exportFigure.export_plot_as_html(fig_4, '008_kmeans_rank_selection_mut')

fig_1.show()
fig_2.show()
fig_3.show()
fig_4.show()






