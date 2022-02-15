from numpy import unique
from sklearn.mixture import GaussianMixture
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
    ('Gaussian', 'GAUSSIAN'),
    ('DBSCAN', 'DBSCAN')
]


# clusters the given data with the select algorithm and display a plot of two features
# instance_list: The data to cluster
# params: Parameters for the clustering algorithms
def cluster(instances_list, params_dict):

    algorithm = params_dict['cluster_algorithm']

    # select clustering algorithm
    if algorithm == "KMEANS":
        model = KMeans(random_state=params_dict['seed'], n_clusters=params_dict['n_clusters_k_means'])
    elif algorithm == "AFFINITY":
        model = AffinityPropagation(random_state=params_dict['seed'], damping=params_dict['damping_aff'],
                                    preference=params_dict['preference_aff'], affinity=params_dict['affinity_aff'])
    elif algorithm == "MEANSHIFT":
        model = MeanShift(bandwidth=params_dict['bandwidth_mean'])
    elif algorithm == "SPECTRAL":
        model = SpectralClustering(random_state=params_dict['seed'], n_clusters=params_dict['n_clusters_spectral'])
    elif algorithm == "AGGLOMERATIVE":
        model = AgglomerativeClustering(n_clusters=params_dict['n_clusters_agg'], affinity=params_dict['affinity_agg'],
                                        linkage=params_dict['linkage_agg'], distance_threshold=params_dict['distance_threshold'])
    elif algorithm == "OPTICS":
        model = OPTICS(min_samples=params_dict['min_samples_opt'], min_cluster_size=params_dict['min_clusters_opt'])
    elif algorithm == "GAUSSIAN":
        model = GaussianMixture(random_state=params_dict['seed'], n_components=params_dict['n_components_gauss'])
    elif algorithm == "BIRCH":
        model = Birch(threshold=params_dict['threshold_birch'], branching_factor=params_dict['branching_factor_birch'],
                      n_clusters=params_dict['n_clusters_birch'])
    else:  # algorithm == "DBSCAN":
        model = DBSCAN(eps=params_dict['eps_dbscan'], min_samples=params_dict['min_samples_dbscan'])

    # fit model and extract clusters
    model.fit(instances_list)
    if algorithm == "GAUSSIAN":
        yhat = model.predict(instances_list)
    else:
        yhat = model.labels_
    clusters = unique(yhat)

    # return clusters and the mapping of each instance to the cluster
    return clusters, yhat
