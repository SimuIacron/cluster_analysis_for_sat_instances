import json
from datetime import datetime
import os

from DataFormats.InputData import InputDataFeatureSelection, InputDataCluster, InputDataScaling


def export_json(dataset, cluster_params: InputDataCluster, feature_selection_params: InputDataFeatureSelection,
                scaling_params: InputDataScaling):
    misc_dict = {'dataset_selection': dataset}

    cluster_dict = {key: value for key, value in cluster_params.__dict__.items() if
                    not key.startswith('__') and not callable(key)}
    feature_selection_dict = {key: value for key, value in feature_selection_params.__dict__.items() if
                              not key.startswith('__') and not callable(key)}
    scaling_dict = {key: value for key, value in scaling_params.__dict__.items() if
                    not key.startswith('__') and not callable(key)}

    dict_params = {'misc': misc_dict, 'cluster_params': cluster_dict,
                   'feature_selection_params': feature_selection_dict, 'scaling_dict': scaling_dict}

    dict_final = {'params': dict_params}

    now = datetime.now()
    dt_string = now.strftime("%y-%m-%d_%H-%M-%S")

    with open(os.environ['JSONPATH'] + dt_string, 'w') as outfile:
        json.dump(dict_final, outfile)


def convert_bytes(data):
    dict_final = json.loads(data.decode('utf-8'))
    dict_params = dict_final['params']
    dataset = dict_params['misc']['dataset_selection']
    cluster_dict = dict_params['cluster_params']
    feature_selection_dict = dict_params['feature_selection_params']
    scaling_dict = dict_params['scaling_dict']

    cluster_params = \
        InputDataCluster(cluster_algorithm=cluster_dict['cluster_algorithm'], seed=cluster_dict['seed'],
                         n_clusters_k_means=cluster_dict['n_clusters_k_means'],
                         damping_aff=cluster_dict['damping_aff'], preference_aff=cluster_dict['preference_aff'], affinity_aff=cluster_dict['affinity_aff'],
                         bandwidth_mean=cluster_dict['bandwidth_mean'],
                         n_clusters_spectral=cluster_dict['n_clusters_spectral'],
                         n_clusters_agg=cluster_dict['n_clusters_agg'], affinity_agg=cluster_dict['affinity_agg'], linkage_agg=cluster_dict['linkage_agg'],
                         distance_threshold=cluster_dict['distance_threshold'],
                         min_samples_opt=cluster_dict['min_samples_opt'], min_clusters_opt=cluster_dict['min_clusters_opt'],
                         n_components_gauss=cluster_dict['n_components_gauss'],
                         threshold_birch=cluster_dict['threshold_birch'], branching_factor_birch=cluster_dict['branching_factor_birch'],
                         n_clusters_birch=cluster_dict['n_clusters_birch'],
                         eps_dbscan=cluster_dict['eps_dbscan'], min_samples_dbscan=cluster_dict['min_samples_dbscan'])

    scaling_params = InputDataScaling(scaling_algorithm=scaling_dict['scaling_algorithm'],
                                      scaling_technique=scaling_dict['scaling_technique'],
                                      scaling_k_best=scaling_dict['scaling_k_best'])

    selection_params = \
        InputDataFeatureSelection(selection_algorithm=feature_selection_dict['selection_algorithm'],
                                  seed=feature_selection_dict['seed'],
                                  n_features_pca=feature_selection_dict['n_features_pca'],
                                  variance_var=feature_selection_dict['variance_var'],
                                  n_components_sparse=feature_selection_dict['n_components_sparse'],
                                  n_components_gaussian=feature_selection_dict['n_components_gaussian'])

    return dataset, cluster_params, selection_params, scaling_params


def convert_bytes_view(data):
    dict_final = json.loads(data.decode('utf-8'))
    dict_params = dict_final['params']
    dataset = dict_params['misc']['dataset_selection']
    cluster_dict = dict_params['cluster_params']
    feature_selection_dict = dict_params['feature_selection_params']
    scaling_dict = dict_params['scaling_dict']

    if feature_selection_dict['n_components_sparse'] == 'auto':
        n_components_sparse = -1
    else:
        n_components_sparse = feature_selection_dict['n_components_sparse']

    if feature_selection_dict['n_components_gaussian'] == 'auto':
        n_components_gaussian = -1
    else:
        n_components_gaussian = feature_selection_dict['n_components_gaussian']

    if cluster_dict['preference_aff'] is None:
        preference_aff = -1
    else:
        preference_aff = cluster_dict['preference_aff']

    if cluster_dict['bandwidth_mean'] is None:
        bandwidth_mean = -1
    else:
        bandwidth_mean = cluster_dict['bandwidth_mean']

    if cluster_dict['distance_threshold'] is None:
        distance_threshold = -1
    else:
        distance_threshold = cluster_dict['distance_threshold']

    return [
        dataset,

        scaling_dict['scaling_algorithm'],
        scaling_dict['scaling_technique'],
        scaling_dict['scaling_k_best'],

        feature_selection_dict['selection_algorithm'],
        feature_selection_dict['seed'],
        feature_selection_dict['n_features_pca'],
        feature_selection_dict['variance_var'],
        n_components_sparse,
        n_components_gaussian,

        cluster_dict['cluster_algorithm'],
        cluster_dict['seed'],
        cluster_dict['n_clusters_k_means'],
        cluster_dict['damping_aff'],
        preference_aff,
        cluster_dict['affinity_aff'],
        bandwidth_mean,
        cluster_dict['n_clusters_spectral'],
        cluster_dict['n_clusters_agg'],
        cluster_dict['affinity_agg'],
        cluster_dict['linkage_agg'],
        distance_threshold,
        cluster_dict['min_samples_opt'],
        cluster_dict['min_clusters_opt'],
        cluster_dict['n_components_gauss'],
        cluster_dict['threshold_birch'],
        cluster_dict['branching_factor_birch'],
        cluster_dict['n_clusters_birch'],
        cluster_dict['eps_dbscan'],
        cluster_dict['min_samples_dbscan']
    ]