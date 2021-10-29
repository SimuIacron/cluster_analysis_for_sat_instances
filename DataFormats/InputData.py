class InputDataCluster:

    def __init__(self, cluster_algorithm='KMEANS', n_clusters=5, eps=0.2, min_samples=10):
        self.cluster_algorithm = cluster_algorithm
        self.n_clusters = n_clusters
        self.eps = eps
        self.min_samples = min_samples


class InputDataFeatureSelection:

    def __init__(self, selection_algorithm='PCA', n_features=0, variance=0.8):
        self.selection_algorithm = selection_algorithm
        self.n_features = n_features
        self.variance = variance


class InputDataScaling:

    def __init__(self, scaling_algorithm='SCALEMINUSPLUS1'):
        self.scaling_algorithm = scaling_algorithm
