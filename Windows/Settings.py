import pandas as pd
import plotly.express as px
from dash import html, Input, Output, dcc, State
import dash_daq.NumericInput as daq
from numpy import argmin

import DatabaseReader
import util
from DataAnalysis import scaling, feature_reduction, clustering, evaluation
from DataFormats.InputData import InputDataScaling, InputDataCluster, InputDataFeatureSelection
from DataFormats.DbInstance import DbInstance
from Windows import ClusterStatistics


def create_dropdown(dropdown_data):
    options = []
    for field in dropdown_data:
        options.append({'label': field[0], 'value': field[1]})

    return options


def create_layout():
    return [
        html.H4('Scaling'),
        dcc.Dropdown(
            id='dropdown-scaling-algorithm',
            options=create_dropdown(scaling.SCALINGALGORITHMS),
            value=create_dropdown(scaling.SCALINGALGORITHMS)[0]['value']
        ),
        html.H4('Feature Filtering'),
        dcc.Dropdown(
            id='dropdown-feature-selection-algorithm',
            options=create_dropdown(feature_reduction.FEATURESELECTIONALGORITHMS),
            value=create_dropdown(feature_reduction.FEATURESELECTIONALGORITHMS)[0]['value']
        ),
        html.Label('Remaining Features (0 means open) (PCA)'),
        daq.NumericInput(
            id='numeric-input-features-selection',
            value=10,
            min=0
        ),
        html.Label('Variance (Variance)'),
        dcc.Slider(
            id='slider-variance',
            min=0.1,
            max=1,
            step=0.01,
            value=0.8,
            tooltip={"placement": "bottom", "always_visible": False},
        ),
        html.H4('Clustering'),
        dcc.Dropdown(
            id='dropdown-cluster-algorithm',
            options=create_dropdown(clustering.CLUSTERALGORITHMS),
            value=create_dropdown(clustering.CLUSTERALGORITHMS)[0]['value']
        ),
        html.Label('Cluster amounts (K-Means, Gaussian)'),
        daq.NumericInput(
            id='numeric-input-n-cluster',
            value=5,
            min=1
        ),
        html.Label('Eps (DBSCAN)'),
        dcc.Slider(
            id='slider-eps',
            min=0.1,
            max=10,
            step=0.1,
            value=1,
            tooltip={"placement": "bottom", "always_visible": False},
        ),
        html.Label('Min Samples (DBSCAN)'),
        daq.NumericInput(
            id='numeric-input-min-samples',
            value=5,
            min=1
        ),
        html.Button('Calculate', id='submit-val', n_clicks=0)
    ]


def register_callback(app, db_instance: DbInstance):
    @app.callback(
        [Output('clustering-graph-1', 'figure'),
         Output('clustering-graph-2', 'figure'),
         Output('clustering-graph-3', 'figure'),
         Output('div-cluster-statistics', 'children')],
        [Input('submit-val', 'n_clicks')],
        [State('dropdown-scaling-algorithm', 'value'),
         State('dropdown-feature-selection-algorithm', 'value'),
         State('dropdown-cluster-algorithm', 'value'),
         State('numeric-input-n-cluster', 'value'),
         State('slider-eps', 'value'),
         State('numeric-input-min-samples', 'value'),
         State('numeric-input-features-selection', 'value'),
         State('slider-variance', 'value')]
    )
    def update_output(n_clicks, scaling_value, feature_reduction_value, cluster_value, n_clusters, eps, min_samples,
                      n_features, variance):
        input_data_cluster = InputDataCluster(cluster_algorithm=cluster_value, n_clusters=n_clusters, eps=eps,
                                              min_samples=min_samples)
        input_data_scaling = InputDataScaling(scaling_algorithm=scaling_value)
        input_data_feature_selection = InputDataFeatureSelection(selection_algorithm=feature_reduction_value,
                                                                 n_features=n_features, variance=variance)

        clusters, yhat, reduced_instance_list, instances_list_s = run_clustering(db_instance, input_data_cluster,
                                                                                 input_data_scaling,
                                                                                 input_data_feature_selection)
        return [
            evaluation.clusters_scatter_plot(yhat, reduced_instance_list, db_instance.solver_wh,
                                             DatabaseReader.FEATURES_SOLVER),
            evaluation.clusters_family_amount(clusters, yhat, db_instance.family_wh),
            evaluation.clusters_timeout_amount(clusters, yhat, DatabaseReader.TIMEOUT, db_instance.solver_wh),
            ClusterStatistics.create_layout(clusters, yhat, db_instance)
        ]


def run_clustering(db_instance: DbInstance, input_data_cluster: InputDataCluster, input_data_scaling: InputDataScaling,
                   input_data_feature_selection: InputDataFeatureSelection):
    print("Start scaling...")
    instances_list_s = scaling.scaling(db_instance.instances_wh, algorithm=input_data_scaling.scaling_algorithm)

    print("Scaling finished")

    # reduce dimensions
    reduced_instance_list = \
        feature_reduction.feature_reduction(instances_list_s,
                                            algorithm=input_data_feature_selection.selection_algorithm,
                                            features=input_data_feature_selection.n_features,
                                            variance=input_data_feature_selection.variance)

    print('Remaining features: ' + str(len(reduced_instance_list[0])))
    print("Starting clustering...")

    # clustering
    (clusters, yhat) = clustering.cluster(reduced_instance_list, algorithm=input_data_cluster.cluster_algorithm,
                                          n_clusters=input_data_cluster.n_clusters, eps=input_data_cluster.eps,
                                          min_samples=input_data_cluster.min_samples)
    print("Clustering finished")

    return clusters, yhat, reduced_instance_list, instances_list_s
