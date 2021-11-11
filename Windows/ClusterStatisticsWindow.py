from dash import dcc, html

import DatabaseReader
from DataAnalysis import evaluation
from DataFormats.DbInstance import DbInstance


# creates for each cluster that exists two graphs
def create_layout(clusters, yhat, db_instance: DbInstance):
    graphs = []
    for cluster in clusters:

        graph1 = dcc.Graph(
            style={'height': 800},
            figure=evaluation.cluster_statistics(cluster, yhat, db_instance.solver_wh, DatabaseReader.FEATURES_SOLVER)
        )

        graph3 = dcc.Graph(
            style={'height': 800},
            figure=evaluation.cluster_family_amount(cluster, yhat, db_instance.family_wh)
        )

        graphs.append(graph1)
        graphs.append(graph3)

    return graphs
