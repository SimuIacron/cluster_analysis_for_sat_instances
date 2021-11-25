from sklearn.metrics import adjusted_mutual_info_score
import plotly.graph_objects as go
import numpy as np

import exportFigure
from DataAnalysis import feature_selection, scaling, clustering
from DataAnalysis.Evaluation import scoring_util
from DataFormats.DbInstance import DbInstance
from DataFormats.InputData import InputDataCluster, InputDataScaling, InputDataFeatureSelection

from itertools import combinations

input = ['base', 'gate', 'solver']

output = sum([list(map(list, combinations(input, i))) for i in range(len(input) + 1)], [])

db_instance = DbInstance()

family_int = scoring_util.convert_families_to_int(db_instance.family_wh)
solver_int = scoring_util.convert_best_solver_int(db_instance)

surface_family = []
surface_solver = []


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

    reduced_instance_list = feature_selection.feature_selection(
            db_instance.dataset_wh, db_instance.dataset_f, db_instance.solver_wh, input_data_feature_selection
        )

    instances_list_s = scaling.scaling(reduced_instance_list, db_instance.dataset_f, input_data_scaling)

    mutual_info_solver_list = []
    mutual_info_family_list = []
    eps_range = np.arange(0.1, 2, 0.1)
    min_sample_range = range(1, 30)

    for samples in min_sample_range:
        current_eps_family = []
        current_eps_solver = []
        for eps in eps_range:

            input_data_cluster = InputDataCluster(
                cluster_algorithm='DBSCAN',
                seed=0,
                eps_dbscan=eps,
                min_samples_dbscan=samples
            )

            # clustering
            (clusters, yhat) = clustering.cluster(instances_list_s, input_data_cluster)

            mutual_info = adjusted_mutual_info_score(solver_int, yhat)
            current_eps_solver.append(mutual_info)
            mutual_info = adjusted_mutual_info_score(family_int, yhat)
            current_eps_family.append(mutual_info)

        mutual_info_solver_list.append(current_eps_solver)
        mutual_info_family_list.append(current_eps_family)

    surface_family.append(go.Surface(z=mutual_info_family_list, x=list(min_sample_range), y=eps_range, opacity=0.7, name=" ".join(str(x) for x in comb)))
    surface_solver.append(go.Surface(z=mutual_info_solver_list, x=list(min_sample_range), y=eps_range, opacity=0.7, name=" ".join(str(x) for x in comb)))


fig_family = go.Figure(layout=go.Layout(
        title=go.layout.Title(text="Family")
    ),
    data=surface_family
)
fig_solver = go.Figure(layout=go.Layout(
        title=go.layout.Title(text="Solver")
    ),
    data=surface_solver
)

exportFigure.export_plot_as_html(fig_family, '003_dbscan_mut_family_eps_samples')
exportFigure.export_plot_as_html(fig_solver, '003_dbscan_mut_solver_eps_samples')

fig_family.show()
fig_solver.show()






