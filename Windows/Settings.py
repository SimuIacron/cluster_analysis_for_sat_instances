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

        html.Button('Calculate', id='submit-val', n_clicks=0),

        html.H2('Scaling'),
        dcc.Dropdown(
            id='dropdown-scaling-algorithm',
            options=create_dropdown(scaling.SCALINGALGORITHMS),
            value=create_dropdown(scaling.SCALINGALGORITHMS)[0]['value']
        ),
        html.H2('Feature Filtering'),
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

        # clustering
        html.H2('Clustering'),
        dcc.Dropdown(
            id='dropdown-cluster-algorithm',
            options=create_dropdown(clustering.CLUSTERALGORITHMS),
            value=create_dropdown(clustering.CLUSTERALGORITHMS)[0]['value']
        ),
        html.Label('Seed'),
        daq.NumericInput(
            id='numeric-input-seed',
            value=0,
            min=0
        ),

        html.H4('K-Means'),
        html.Label('Cluster amounts'),
        daq.NumericInput(
            id='numeric-input-n-cluster-k-means',
            value=5,
            min=1
        ),

        html.H4('Affinity'),
        html.Label('Damping'),
        dcc.Slider(
            id='slider-damping-aff',
            min=0.5,
            max=1,
            step=0.01,
            value=0.5,
            tooltip={"placement": "bottom", "always_visible": False},
        ),
        html.Label('Preference'),
        daq.NumericInput(
            id='numeric-input-preference-aff',
            value=-1,
            min=-1
        ),
        html.Label('Affinity'),
        dcc.Dropdown(
            id='dropdown-affinity-aff',
            options=create_dropdown([('euclidean', 'euclidean'), ('precomputed', 'precomputed')]),
            value='euclidean'
        ),

        html.H4('MeanShift'),
        html.Label('Bandwidth'),
        dcc.Slider(
            id='slider-bandwidth-mean',
            min=-1,
            max=25,
            step=0.1,
            value=-1,
            tooltip={"placement": "bottom", "always_visible": False},
        ),

        html.H4('Spectral Clustering'),
        html.Label('Cluster Amounts'),
        daq.NumericInput(
            id='numeric-input-n-cluster-spectral',
            value=5,
            min=1
        ),


        html.H4('Agglomerative Clustering'),
        html.Label('Cluster Amounts'),
        daq.NumericInput(
            id='numeric-input-n-cluster-agg',
            value=2,
            min=1
        ),
        html.Label('Affinity'),
        dcc.Dropdown(
            id='dropdown-affinity-agg',
            options=create_dropdown([('euclidean', 'euclidean'), ('l1', 'l1'), ('l2', 'l2'), ('manhattan', 'manhattan'),
                                     ('cosine', 'cosine')]),
            value='euclidean'
        ),
        html.Label('Linkage'),
        dcc.Dropdown(
            id='dropdown-linkage-agg',
            options=create_dropdown(
                [('ward', 'ward'), ('complete', 'complete'), ('average', 'average'), ('single', 'single')]),
            value='ward'
        ),
        html.Label('Distance Threshold'),
        dcc.Slider(
            id='slider-distance-threshold-agg',
            min=-1,
            max=25,
            step=0.1,
            value=-1,
            tooltip={"placement": "bottom", "always_visible": False},
        ),

        html.H4('Optics'),
        html.Label('Min samples'),
        daq.NumericInput(
            id='numeric-input-min-samples-opt',
            value=5,
            min=1
        ),
        html.Label('Min clusters'),
        daq.NumericInput(
            id='numeric-input-min-clusters-opt',
            value=3,
            min=1
        ),

        html.H4('Gaussian'),
        html.Label('Number of mixture components'),
        daq.NumericInput(
            id='numeric-input-components-gauss',
            value=1,
            min=1
        ),

        html.H4('Birch'),
        html.Label('Threshold'),
        dcc.Slider(
            id='slider-threshold-birch',
            min=0.1,
            max=10,
            step=0.01,
            value=0.5,
            tooltip={"placement": "bottom", "always_visible": False},
        ),
        html.Label('Branching Factor'),
        daq.NumericInput(
            id='numeric-input-branching-factor-birch',
            value=50,
            min=2
        ),
        html.Label('Cluster Amount'),
        daq.NumericInput(
            id='numeric-input-n-clusters-birch',
            value=3,
            min=1
        ),

        html.H4('DBSCAN'),
        html.Label('Eps (DBSCAN)'),
        dcc.Slider(
            id='slider-eps-dbscan',
            min=0.1,
            max=10,
            step=0.1,
            value=1,
            tooltip={"placement": "bottom", "always_visible": False},
        ),
        html.Label('Min Samples (DBSCAN)'),
        daq.NumericInput(
            id='numeric-input-min-samples-dbscan',
            value=5,
            min=1
        )
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
         State('numeric-input-features-selection', 'value'),
         State('slider-variance', 'value'),

         State('dropdown-cluster-algorithm', 'value'),
         State('numeric-input-seed', 'value'),
         State('numeric-input-n-cluster-k-means', 'value'),
         State('slider-damping-aff', 'value'),
         State('numeric-input-preference-aff', 'value'),
         State('dropdown-affinity-aff', 'value'),
         State('slider-bandwidth-mean', 'value'),
         State('numeric-input-n-cluster-spectral', 'value'),
         State('numeric-input-n-cluster-agg', 'value'),
         State('dropdown-affinity-agg', 'value'),
         State('dropdown-linkage-agg', 'value'),
         State('slider-distance-threshold-agg', 'value'),
         State('numeric-input-min-samples-opt', 'value'),
         State('numeric-input-min-clusters-opt', 'value'),
         State('numeric-input-components-gauss', 'value'),
         State('slider-threshold-birch', 'value'),
         State('numeric-input-branching-factor-birch', 'value'),
         State('numeric-input-n-clusters-birch', 'value'),
         State('slider-eps-dbscan', 'value'),
         State('numeric-input-min-samples-dbscan', 'value')
         ]
    )
    def update_output(n_clicks,
                      scaling_value,
                      feature_reduction_value, n_features, variance,
                      cluster_value, seed,
                      n_clusters_k_means,
                      damping_aff, preference_aff, affinity_aff,
                      bandwidth_mean,
                      n_clusters_spectral,
                      n_clusters_agg, affinity_agg, linkage_agg, distance_threshold,
                      min_samples_opt, min_clusters_opt,
                      n_components_gauss,
                      threshold_birch, branching_factor_birch, n_clusters_birch,
                      eps_dbscan, min_samples_dbscan):
        input_data_cluster = InputDataCluster(cluster_algorithm=cluster_value, seed=seed,
                                              n_clusters_k_means=n_clusters_k_means,
                                              damping_aff=damping_aff, preference_aff=preference_aff, affinity_aff=affinity_aff,
                                              bandwidth_mean=bandwidth_mean,
                                              n_clusters_spectral=n_clusters_spectral,
                                              n_clusters_agg=n_clusters_agg, affinity_agg=affinity_agg, linkage_agg=linkage_agg, distance_threshold=distance_threshold,
                                              min_samples_opt=min_samples_opt, min_clusters_opt=min_clusters_opt,
                                              n_components_gauss=n_components_gauss,
                                              threshold_birch=threshold_birch, branching_factor_birch=branching_factor_birch, n_clusters_birch=n_clusters_birch,
                                              eps_dbscan=eps_dbscan, min_samples_dbscan=min_samples_dbscan)
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


def run_clustering(db_instance: DbInstance, input_data_cluster: InputDataCluster,
                   input_data_scaling: InputDataScaling,
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
    (clusters, yhat) = clustering.cluster(reduced_instance_list, input_data_cluster)
    print("Clustering finished")

    return clusters, yhat, reduced_instance_list, instances_list_s
