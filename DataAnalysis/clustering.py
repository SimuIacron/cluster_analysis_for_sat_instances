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
# axis1: feature that is used as the x axis in the plot
# axis2: feature that is used as the y axis in the plot
# algorithm: The cluster algorithm
def cluster(instances_list, algorithm="DBSCAN", n_clusters=5, eps=1, min_samples=10):
    print("Starting clustering...")

    # select clustering algorithm
    if algorithm == "KMEANS":
        model = KMeans(n_clusters=n_clusters)
    elif algorithm == "AFFINITY":
        model = AffinityPropagation()
    elif algorithm == "MEANSHIFT":
        model = MeanShift()
    elif algorithm == "SPECTRAL":
        model = SpectralClustering()
    elif algorithm == "AGGLOMERATIVE":
        model = AgglomerativeClustering()
    elif algorithm == "OPTICS":
        model = OPTICS()
    elif algorithm == "BIRCH":
        model = Birch()
    elif algorithm == "GAUSSIAN":
        model = GaussianMixture(n_components=n_clusters)
    else:  # algorithm == "DBSCAN":
        model = DBSCAN(eps=eps, min_samples=min_samples)

    # fit model and extract clusters
    model.fit(instances_list)
    if algorithm == "GAUSSIAN":
        yhat = model.predict(instances_list)
    else:
        yhat = model.labels_
    clusters = unique(yhat)

    # return clusters and the mapping of each instance to the cluster
    return clusters, yhat

    # fig = pyplot.figure()
    # ax = fig.add_subplot(projection='2d')

    # x_axis = []
    # y_axis = []
    # z_axis = []
    # for cluster in clusters:
    #     for i in range(len(yhat)):
    #        if yhat[i] == cluster:
    #            x_axis.append(instances_list[i][index1])
    #            y_axis.append(instances_list[i][index2])
    #            z_axis.append(instances_list[i][index3])

    #       pyplot.scatter(x_axis, y_axis) #, z_axis)
    # pyplot.show()
