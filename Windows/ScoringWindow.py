
# creates layout with statistics
from dash import html, dcc

from DataAnalysis import evaluation, scoring
from DataFormats.DbInstance import DbInstance


def create_layout(clusters, yhat, db_instance: DbInstance):
    # family_labels = [html.H2('Family scores')]
    # for stats in family_statistics:
    #    family_labels.append(html.Label(stats[0] + ': ' + str(stats[1])))
    #    family_labels.append(html.Br())

    graph1 = dcc.Graph(
        style={'height': 800},
        figure=evaluation.family_score_chart(yhat, db_instance)
    )

    graph2 = dcc.Graph(
        style={'height': 800},
        figure=evaluation.solver_score_cluster(clusters, yhat, db_instance)
    )

    graph3 = dcc.Graph(
        style={'height': 800},
        figure=evaluation.solver_score_cluster_linear(clusters, yhat, db_instance)
    )

    return [graph1, graph2, graph3]
