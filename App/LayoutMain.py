from dash import html, Output, Input
from dash import dcc

from DataFormats.DbInstance import DbInstance
from Windows import ClusteringGraphWindow, SettingsWindow


# inits a layout with three tabs and calls the window files for the first two
# the third tab is an output which gets filled when the clusters are calculated, because it layout depends
# on the amount of clusters.
def init_layout():
    return [dcc.Tabs([
        dcc.Tab(label='Settings', children=SettingsWindow.create_layout()),
        dcc.Tab(label='Cluster Graph', children=ClusteringGraphWindow.create_layout()),
        dcc.Tab(label='Cluster Statistics', children=[html.Div(id='div-cluster-statistics')]),
        dcc.Tab(label='Scoring', children=[html.Div(id='div-scoring')])
    ])]


# registers the callbacks in the settings tab
def register_callbacks(app, db_instance: DbInstance):
    SettingsWindow.register_callback(app, db_instance)



