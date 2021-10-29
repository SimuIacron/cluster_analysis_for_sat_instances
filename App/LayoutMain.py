from dash import html, Output, Input
from dash import dcc

from DataFormats.DbInstance import DbInstance
from Windows import ClusteringGraph, Settings


def init_layout():
    return [dcc.Tabs([
        dcc.Tab(label='Settings', children=Settings.create_layout()),
        dcc.Tab(label='Cluster Graph', children=ClusteringGraph.create_layout()),
        dcc.Tab(label='Cluster Statistics', children=[html.Div(id='div-cluster-statistics')]),
    ])]


def register_callbacks(app, db_instance: DbInstance):
    Settings.register_callback(app, db_instance)



