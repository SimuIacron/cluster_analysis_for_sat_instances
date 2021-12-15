import base64

import dash.exceptions
from dash import html, Input, Output, dcc, State
import dash_daq.NumericInput as daq

from util_scripts import JsonExport, DatabaseReader
from DataAnalysis import scaling, feature_selection, clustering, evaluation
from DataFormats.InputData import InputDataScaling, InputDataCluster, InputDataFeatureSelection
from DataFormats.DbInstance import DbInstance
from Dash.Windows import ClusterStatisticsWindow, ScoringWindow


def create_dropdown(dropdown_data):
    options = []
    for field in dropdown_data:
        options.append({'label': field[0], 'value': field[1]})

    return options


def create_layout():
    return [

        html.Button('Calculate', id='submit-val', n_clicks=0),

        dcc.Checklist(
            id='checkbox-export-json',
            options=[{'label': 'Export settings as json', 'value': 'export_json'}],
            value=['']
        ),

        dcc.Upload(
            id='json-upload',
            children=html.Div(
                ["Upload json file to import settings."]
            ),
            style={
                "width": "100%",
                "height": "60px",
                "lineHeight": "60px",
                "borderWidth": "1px",
                "borderStyle": "dashed",
                "borderRadius": "5px",
                "textAlign": "center",
                "margin": "10px",
            }
        ),

        html.Label('Cluster Dataset'),
        dcc.Checklist(
            id='checkbox-dataset',
            options=[
                {'label': 'Base', 'value': 'base'},
                {'label': 'Gate', 'value': 'gate'},
                {'label': 'Solver', 'value': 'solver'}
            ],
            value=['base', 'gate']
        ),

        html.H2('Scaling'),
        dcc.Dropdown(
            id='dropdown-scaling-algorithm',
            options=create_dropdown(scaling.SCALINGALGORITHMS),
            value=create_dropdown(scaling.SCALINGALGORITHMS)[0]['value']
        ),

        html.H4('Scale to [-1,1]'),
        html.Label('Scaling technique'),
        dcc.Dropdown(
            id='dropdown-scaling-technique',
            options=create_dropdown(scaling.SCALINGTECHNIQUES),
            value=create_dropdown(scaling.SCALINGTECHNIQUES)[0]['value']
        ),
        html.Label('K'),
        daq.NumericInput(
            id='numeric-input-k-best',
            value=3,
            min=1,
            max=1000
        ),

        html.H2('Feature Filtering'),
        dcc.Dropdown(
            id='dropdown-feature-selection-algorithm',
            options=create_dropdown(feature_selection.FEATURESELECTIONALGORITHMS),
            value=create_dropdown(feature_selection.FEATURESELECTIONALGORITHMS)[0]['value']
        ),

        html.Label('Seed Filtering'),
        daq.NumericInput(
            id='numeric-input-seed-filtering',
            value=0,
            min=0,
            max=1000
        ),

        html.H4('Variance'),
        html.Label('Variance'),
        dcc.Slider(
            id='slider-variance-var',
            min=0.1,
            max=1,
            step=0.01,
            value=0.8,
            tooltip={"placement": "bottom", "always_visible": False},
        ),

        html.Label('Variance ignore'),
        dcc.Dropdown(
            id='dropdown-feature-selection-var-ignore',
            options=create_dropdown(feature_selection.FEATURESELECTIONIGNORE),
            value=create_dropdown(feature_selection.FEATURESELECTIONIGNORE)[0]['value']
        ),

        html.H4('PCA'),
        html.Label('Remaining Features (0 means open)'),
        daq.NumericInput(
            id='numeric-input-features-selection-pca',
            value=10,
            min=0,
            max=1000
        ),

        html.H4('Sparse'),
        html.Label('Component amount (-1 is auto)'),
        daq.NumericInput(
            id='numeric-input-components-sparse',
            value=-1,
            min=-1,
            max=1000
        ),

        html.H4('Gaussian'),
        html.Label('Component amount (-1 is auto)'),
        daq.NumericInput(
            id='numeric-input-components-gaussian',
            value=-1,
            min=-1,
            max=1000
        ),

        # clustering
        html.H2('Clustering'),
        dcc.Dropdown(
            id='dropdown-cluster-algorithm',
            options=create_dropdown(clustering.CLUSTERALGORITHMS),
            value=create_dropdown(clustering.CLUSTERALGORITHMS)[0]['value']
        ),

        html.Label('Seed Clustering'),
        daq.NumericInput(
            id='numeric-input-seed-clustering',
            value=0,
            min=0,
            max=1000
        ),

        html.H4('K-Means'),
        html.Label('Cluster amounts'),
        daq.NumericInput(
            id='numeric-input-n-cluster-k-means',
            value=5,
            min=1,
            max=1000
        ),

        html.H4('Affinity'),
        html.Label('Damping: Damps the responsibility and availability messages to avoid numerical oscillations'),
        dcc.Slider(
            id='slider-damping-aff',
            min=0.5,
            max=1,
            step=0.01,
            value=0.5,
            tooltip={"placement": "bottom", "always_visible": False},
        ),
        html.Label('Preference (-1 means None): Controls how many exemplars are used'),
        daq.NumericInput(
            id='numeric-input-preference-aff',
            value=-1,
            min=-1,
            max=1000
        ),
        html.Label('Affinity'),
        dcc.Dropdown(
            id='dropdown-affinity-aff',
            options=create_dropdown([('euclidean', 'euclidean'), ('precomputed', 'precomputed')]),
            value='euclidean'
        ),

        html.H4('MeanShift'),
        html.Label('Bandwidth (-1 means it gets estimated): Dictates the size of the region to search through'),
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
            min=1,
            max=1000
        ),

        html.H4('Agglomerative Clustering'),
        html.Label('Cluster Amounts'),
        daq.NumericInput(
            id='numeric-input-n-cluster-agg',
            value=2,
            min=1,
            max=1000
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
                [('ward (minimizes the sum squared difference within all clusters)', 'ward'),
                 ('complete (minimizes the maximum distance between observations of pairs of cluster)s', 'complete'),
                 ('average (minimizes the average of the distance between all observations of pairs of clusters)',
                  'average'),
                 ('single (minimizes the distance between the closest observation of pairs of clusters)', 'single')]),
            value='ward'
        ),
        html.Label('Distance Threshold (-1 means None)'),
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
            min=1,
            max=1000
        ),
        html.Label('Min clusters'),
        daq.NumericInput(
            id='numeric-input-min-clusters-opt',
            value=3,
            min=1,
            max=1000
        ),

        html.H4('Gaussian'),
        html.Label('Number of mixture components'),
        daq.NumericInput(
            id='numeric-input-components-gauss',
            value=1,
            min=1,
            max=1000
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
            min=2,
            max=1000
        ),
        html.Label('Cluster Amount'),
        daq.NumericInput(
            id='numeric-input-n-clusters-birch',
            value=3,
            min=1,
            max=1000
        ),

        html.H4('DBSCAN'),
        html.Label('Eps (DBSCAN): In which area we check for other samples around the selected sample'),
        dcc.Slider(
            id='slider-eps-dbscan',
            min=0.1,
            max=10,
            step=0.1,
            value=1,
            tooltip={"placement": "bottom", "always_visible": False},
        ),
        html.Label(
            'Min Samples (DBSCAN): How many points samples have to be around the selected sample to be a core-sample'),
        daq.NumericInput(
            id='numeric-input-min-samples-dbscan',
            value=5,
            min=1,
            max=1000
        )
    ]


def register_callback(app, db_instance: DbInstance):
    # when calculation button (Input) is pressed read in all parameters (State), calculate the clusters
    # and output the graphs and windows (Output)
    @app.callback(
        [Output('clustering-graph-1', 'figure'),
         Output('clustering-graph-2', 'figure'),
         Output('clustering-graph-3', 'figure'),
         Output('div-cluster-statistics', 'children'),
         Output('div-scoring', 'children')],
        [Input('submit-val', 'n_clicks')],
        [State('checkbox-export-json', 'value'),
         State('checkbox-dataset', 'value'),

         State('dropdown-scaling-algorithm', 'value'),
         State('dropdown-scaling-technique', 'value'),
         State('numeric-input-k-best', 'value'),

         State('dropdown-feature-selection-algorithm', 'value'),
         State('numeric-input-seed-filtering', 'value'),
         State('numeric-input-features-selection-pca', 'value'),
         State('slider-variance-var', 'value'),
         State('dropdown-feature-selection-var-ignore', 'value'),
         State('numeric-input-components-sparse', 'value'),
         State('numeric-input-components-gaussian', 'value'),

         State('dropdown-cluster-algorithm', 'value'),
         State('numeric-input-seed-clustering', 'value'),
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
                      export_json,
                      dataset_selection,

                      scaling_value,
                      scaling_technique,
                      scaling_k_best,

                      feature_reduction_value,
                      seed_filtering,
                      n_features_pca,
                      variance_var,
                      ignore_var,
                      n_components_sparse,
                      n_components_gaussian,

                      cluster_value,
                      seed_clustering,
                      n_clusters_k_means,
                      damping_aff, preference_aff, affinity_aff,
                      bandwidth_mean,
                      n_clusters_spectral,
                      n_clusters_agg, affinity_agg, linkage_agg, distance_threshold,
                      min_samples_opt, min_clusters_opt,
                      n_components_gauss,
                      threshold_birch, branching_factor_birch, n_clusters_birch,
                      eps_dbscan, min_samples_dbscan):

        # cluster params
        input_data_cluster = \
            InputDataCluster(cluster_algorithm=cluster_value, seed=seed_clustering,
                             n_clusters_k_means=n_clusters_k_means,
                             damping_aff=damping_aff, preference_aff=preference_aff, affinity_aff=affinity_aff,
                             bandwidth_mean=bandwidth_mean,
                             n_clusters_spectral=n_clusters_spectral,
                             n_clusters_agg=n_clusters_agg, affinity_agg=affinity_agg, linkage_agg=linkage_agg,
                             distance_threshold=distance_threshold,
                             min_samples_opt=min_samples_opt, min_clusters_opt=min_clusters_opt,
                             n_components_gauss=n_components_gauss,
                             threshold_birch=threshold_birch, branching_factor_birch=branching_factor_birch,
                             n_clusters_birch=n_clusters_birch,
                             eps_dbscan=eps_dbscan, min_samples_dbscan=min_samples_dbscan)

        # scaling params
        input_data_scaling = InputDataScaling(scaling_algorithm=scaling_value, scaling_technique=scaling_technique,
                                              scaling_k_best=scaling_k_best)

        # feature selection params
        input_data_feature_selection = \
            InputDataFeatureSelection(selection_algorithm=feature_reduction_value,
                                      seed=seed_filtering,
                                      n_features_pca=n_features_pca,
                                      variance_var=variance_var,
                                      ignore_var=ignore_var,
                                      n_components_sparse=n_components_sparse,
                                      n_components_gaussian=n_components_gaussian)

        db_instance.generate_dataset(dataset_selection)
        # run algorithms
        clusters, yhat, reduced_instance_list, instances_list_s = \
            run(db_instance, input_data_cluster, input_data_scaling, input_data_feature_selection)
        if 'export_json' in export_json:
            JsonExport.export_json(dataset_selection, input_data_cluster, input_data_feature_selection,
                                   input_data_scaling)

        # return graphs and windows
        return [
            evaluation.clusters_scatter_plot(yhat, reduced_instance_list, db_instance.solver_wh,
                                             DatabaseReader.FEATURES_SOLVER),
            evaluation.clusters_family_amount(clusters, yhat, db_instance.family_wh),
            evaluation.clusters_timeout_amount(clusters, yhat, DatabaseReader.TIMEOUT, db_instance.solver_wh),
            ClusterStatisticsWindow.create_layout(clusters, yhat, db_instance),
            ScoringWindow.create_layout(clusters, yhat, db_instance)
        ]

    @app.callback(
        [Output('checkbox-dataset', 'value'),

         Output('dropdown-scaling-algorithm', 'value'),
         Output('dropdown-scaling-technique', 'value'),
         Output('numeric-input-k-best', 'value'),

         Output('dropdown-feature-selection-algorithm', 'value'),
         Output('numeric-input-seed-filtering', 'value'),
         Output('numeric-input-features-selection-pca', 'value'),
         Output('slider-variance-var', 'value'),
         Output('dropdown-feature-selection-var-ignore', 'value'),
         Output('numeric-input-components-sparse', 'value'),
         Output('numeric-input-components-gaussian', 'value'),

         Output('dropdown-cluster-algorithm', 'value'),
         Output('numeric-input-seed-clustering', 'value'),
         Output('numeric-input-n-cluster-k-means', 'value'),
         Output('slider-damping-aff', 'value'),
         Output('numeric-input-preference-aff', 'value'),
         Output('dropdown-affinity-aff', 'value'),
         Output('slider-bandwidth-mean', 'value'),
         Output('numeric-input-n-cluster-spectral', 'value'),
         Output('numeric-input-n-cluster-agg', 'value'),
         Output('dropdown-affinity-agg', 'value'),
         Output('dropdown-linkage-agg', 'value'),
         Output('slider-distance-threshold-agg', 'value'),
         Output('numeric-input-min-samples-opt', 'value'),
         Output('numeric-input-min-clusters-opt', 'value'),
         Output('numeric-input-components-gauss', 'value'),
         Output('slider-threshold-birch', 'value'),
         Output('numeric-input-branching-factor-birch', 'value'),
         Output('numeric-input-n-clusters-birch', 'value'),
         Output('slider-eps-dbscan', 'value'),
         Output('numeric-input-min-samples-dbscan', 'value')
         ],
        [Input('json-upload', 'contents'),
         Input('json-upload', 'filename')])
    def upload_json_file(contents, filename):
        # KNOWN BUG: Reuploading the same file, only works if different file has been uploaded before

        if contents:
            content_type, content_string = contents.split(",")

            decoded = base64.b64decode(content_string)

            # output here the correct values
            values = JsonExport.convert_bytes_view(decoded)
            print(values)
            return values

        # needs to throw an exception if there is no uploaded file
        # to prevent faulty update on startup of application
        raise dash.exceptions.PreventUpdate()


# runs all algorithms with the data and parameters
def run(db_instance: DbInstance, input_data_cluster: InputDataCluster,
        input_data_scaling: InputDataScaling,
        input_data_feature_selection: InputDataFeatureSelection):
    print('Calculation started')



    # feature reduction
    reduced_instance_list = \
        feature_selection.feature_selection(db_instance.dataset_wh, db_instance.dataset_f, db_instance.solver_wh, input_data_feature_selection)

    # scaling
    instances_list_s = scaling.scaling(reduced_instance_list, db_instance.dataset_f, input_data_scaling)

    # clustering
    (clusters, yhat) = clustering.cluster(instances_list_s, input_data_cluster)

    print('Calculation finished')

    return clusters, yhat, reduced_instance_list, instances_list_s
