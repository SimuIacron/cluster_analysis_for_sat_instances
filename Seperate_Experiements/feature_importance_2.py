from sklearn.metrics import normalized_mutual_info_score

import exportFigure
from DataAnalysis import scaling, feature_selection, clustering, scoring, scoring_util
from DataFormats.DbInstance import DbInstance
from DataFormats.InputData import InputDataCluster, InputDataScaling, InputDataFeatureSelection
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd


def get_feature_from_dataset(feature1, feature2, db_instance: DbInstance):
    index1 = db_instance.dataset_f.index(feature1)
    index2 = db_instance.dataset_f.index(feature2)
    feature_list = []
    for inst in db_instance.dataset_wh:
        feature_list.append([inst[index1], inst[index2]])

    return feature_list


input = ['base', 'gate', 'solver']
db_instance = DbInstance()
db_instance.generate_dataset(input)

family_int = scoring_util.convert_families_to_int(db_instance.family_wh)
solver_int = scoring_util.get_best_solver_int(db_instance)
unsat_sat_int = scoring_util.convert_sat_unsat_to_int(db_instance.result_wh)

plot_name_1 = "011_feature_importance_2_family"
plot_name_2 = "011_feature_importance_2_solver"
plot_name_3 = "011_feature_importance_2_sat"

solver = []
family = []
sat = []
for feature1 in db_instance.dataset_f:
    solver_sub = []
    family_sub = []
    sat_sub = []
    for feature2 in db_instance.dataset_f:

        feature_list = get_feature_from_dataset(feature1, feature2, db_instance)

        input_data_feature_selection = InputDataFeatureSelection(
            selection_algorithm='NONE')

        reduced_instance_list = feature_selection.feature_selection(
            feature_list, [feature1, feature2], db_instance.solver_wh, input_data_feature_selection
        )

        input_data_scaling = InputDataScaling(
            scaling_algorithm='SCALEMINUSPLUS1',
            scaling_technique='SCALE01'
        )

        instances_list_s = scaling.scaling(reduced_instance_list, db_instance.dataset_f, input_data_scaling)

        input_data_cluster = InputDataCluster(
            cluster_algorithm='KMEANS',
            seed=0,
            n_clusters_k_means=3
        )

        (clusters, yhat) = clustering.cluster(instances_list_s, input_data_cluster)

        value = normalized_mutual_info_score(solver_int, yhat)
        solver_sub.append(value)
        value = normalized_mutual_info_score(family_int, yhat)
        family_sub.append(value)
        value = normalized_mutual_info_score(unsat_sat_int, yhat)
        sat_sub.append(value)

    family.append(family_sub)
    solver.append(solver_sub)
    sat.append(sat_sub)


fig1 = px.imshow(family, x=db_instance.dataset_f, y=db_instance.dataset_f)
fig1.update_layout(title=plot_name_1)


fig2 = px.imshow(solver, x=db_instance.dataset_f, y=db_instance.dataset_f)
fig2.update_layout(title=plot_name_2)


fig3 = px.imshow(sat, x=db_instance.dataset_f, y=db_instance.dataset_f)
fig3.update_layout(title=plot_name_3)


exportFigure.export_plot_as_html(fig1, plot_name_1)
exportFigure.export_plot_as_html(fig2, plot_name_2)
exportFigure.export_plot_as_html(fig3, plot_name_3)

fig1.show()
fig2.show()
fig3.show()




