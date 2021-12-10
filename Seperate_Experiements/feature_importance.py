from sklearn.metrics import normalized_mutual_info_score

from util import exportFigure
from DataAnalysis import scaling, feature_selection, clustering
from DataAnalysis.Evaluation import scoring_util
from DataFormats.DbInstance import DbInstance
from DataFormats.InputData import InputDataCluster, InputDataScaling, InputDataFeatureSelection
import plotly.express as px
import pandas as pd


def get_feature_from_dataset(feature, db_instance: DbInstance):
    index = db_instance.dataset_f.index(feature)
    feature_list = []
    for inst in db_instance.dataset_wh:
        feature_list.append([inst[index]])

    return feature_list


input = ['base', 'gate', 'solver']
db_instance = DbInstance()
db_instance.generate_dataset(input)

family_int = scoring_util.convert_families_to_int(db_instance.family_wh)
solver_int = scoring_util.convert_best_solver_int(db_instance)
unsat_sat_int = scoring_util.convert_un_sat_to_int(db_instance.un_sat_wh)

plot_name_1 = "011_feature_importance_family"
plot_name_2 = "011_feature_importance_solver"
plot_name_3 = "011_feature_importance_sat"

solver = []
family = []
sat = []
for feature in db_instance.dataset_f:

    feature_list = get_feature_from_dataset(feature, db_instance)

    input_data_feature_selection = InputDataFeatureSelection(
        selection_algorithm='NONE')

    reduced_instance_list = feature_selection.feature_selection(
        feature_list, [feature], db_instance.solver_wh, input_data_feature_selection
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
    solver.append(value)
    value = normalized_mutual_info_score(family_int, yhat)
    family.append(value)
    value = normalized_mutual_info_score(unsat_sat_int, yhat)
    sat.append(value)

df = pd.DataFrame(dict(features=db_instance.dataset_f, score=family))
fig1 = px.bar(df, x='features', y='score', range_y=[0, 1])
fig1.update_layout(title=plot_name_1)

df = pd.DataFrame(dict(features=db_instance.dataset_f, score=solver))
fig2 = px.bar(df, x='features', y='score', range_y=[0, 1])
fig2.update_layout(title=plot_name_2)

df = pd.DataFrame(dict(features=db_instance.dataset_f, score=sat))
fig3 = px.bar(df, x='features', y='score', range_y=[0, 1])
fig3.update_layout(title=plot_name_3)


exportFigure.export_plot_as_html(fig1, plot_name_1)
exportFigure.export_plot_as_html(fig2, plot_name_2)
exportFigure.export_plot_as_html(fig3, plot_name_3)

fig1.show()
fig2.show()
fig3.show()




