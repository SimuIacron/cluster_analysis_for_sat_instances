
# creates layout with statistics
from dash import html, dcc

from DataAnalysis import evaluation


def create_layout(family_statistics):
    #family_labels = [html.H2('Family scores')]
    #for stats in family_statistics:
    #    family_labels.append(html.Label(stats[0] + ': ' + str(stats[1])))
    #    family_labels.append(html.Br())

    graph1 = dcc.Graph(
        style={'height': 800},
        figure=evaluation.family_score_chart(family_statistics)
    )

    return [graph1]
