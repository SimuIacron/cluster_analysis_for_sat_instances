from numpy import where
from numpy import unique
import numpy as np
from matplotlib import pyplot
import pandas as pd
import plotly.express as px
from sklearn.mixture import GaussianMixture

import util
from sklearn.cluster import DBSCAN, KMeans, AffinityPropagation, MeanShift, SpectralClustering, AgglomerativeClustering, \
    OPTICS, Birch

from DataFormats.InputData import InputDataCluster

CLUSTERALGORITHMS = [
    ('K-Means', 'KMEANS'),
    ('Affintiy Propagation', 'AFFINITY'),
    ('Meanshift', 'MEANSHIFT'),
    ('Spectral Clustering', 'SPECTRAL'),
    ('Agglomerative', 'AGGLOMERATIVE'),
    ('OPTICS', 'OPTICS'),
    ('BIRCH', 'BIRCH'),
    ('Guassian', 'GAUSSIAN'),
    ('DBSCAN', 'DBSCAN')
]


# clusters the given data with the select algorithm and display a plot of two features
# instance_list: The data to cluster
# params: Parameters for the clustering algorithms
def cluster(instances_list, params: InputDataCluster):
    print("Starting clustering...")

    algorithm = params.cluster_algorithm

    # select clustering algorithm
    if algorithm == "KMEANS":
        model = KMeans(random_state=params.seed, n_clusters=params.n_clusters_k_means)
    elif algorithm == "AFFINITY":
        model = AffinityPropagation(random_state=params.seed, damping=params.damping_aff,
                                    preference=params.preference_aff, affinity=params.affinity_aff)
    elif algorithm == "MEANSHIFT":
        model = MeanShift(bandwidth=params.bandwidth_mean)
    elif algorithm == "SPECTRAL":
        model = SpectralClustering(random_state=params.seed, n_clusters=params.n_clusters_spectral)
    elif algorithm == "AGGLOMERATIVE":
        model = AgglomerativeClustering(n_clusters=params.n_clusters_agg, affinity=params.affinity_agg,
                                        linkage=params.linkage_agg, distance_threshold=params.distance_threshold)
    elif algorithm == "OPTICS":
        model = OPTICS(min_samples=params.min_samples_opt, min_cluster_size=params.min_clusters_opt)
    elif algorithm == "GAUSSIAN":
        model = GaussianMixture(random_state=params.seed, n_components=params.n_components_gauss)
    elif algorithm == "BIRCH":
        model = Birch(threshold=params.threshold_birch, branching_factor=params.branching_factor_birch,
                      n_clusters=params.n_clusters_birch)
    else:  # algorithm == "DBSCAN":
        model = DBSCAN(eps=params.eps_dbscan, min_samples=params.min_samples_dbscan)

    # fit model and extract clusters
    model.fit(instances_list)
    if algorithm == "GAUSSIAN":
        yhat = model.predict(instances_list)
    else:
        yhat = model.labels_
    clusters = unique(yhat)

    # return clusters and the mapping of each instance to the cluster
    return clusters, yhat
