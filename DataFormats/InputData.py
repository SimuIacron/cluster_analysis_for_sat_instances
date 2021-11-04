# stores the parameters for the clustering algorithms
class InputDataCluster:

    def __init__(self, cluster_algorithm='KMEANS', seed=0,
                 n_clusters_k_means=5,
                 damping_aff=0.5, preference_aff=None, affinity_aff='euclidian',
                 bandwidth_mean=None,
                 n_clusters_spectral=5,
                 n_clusters_agg=2, affinity_agg='euclidian', linkage_agg='ward',distance_threshold=None,
                 min_samples_opt=5, min_clusters_opt=1,
                 n_components_gauss=1,
                 threshold_birch=0.5, branching_factor_birch=50, n_clusters_birch=3,
                 eps_dbscan=0.2, min_samples_dbscan=10):

        # general
        self.seed = seed
        self.cluster_algorithm = cluster_algorithm

        # k-means
        self.n_clusters_k_means = n_clusters_k_means

        # affinity
        self.damping_aff = damping_aff
        if preference_aff == -1:
            self.preference_aff = None
        else:
            self.preference_aff = preference_aff
        self.affinity_aff = affinity_aff

        # meanshift
        if bandwidth_mean == -1:
            self.bandwidth_mean = None
        else:
            self.bandwidth_mean = bandwidth_mean

        # optics
        self.min_samples_opt = min_samples_opt
        self.min_clusters_opt = min_clusters_opt

        # gaussian
        self.n_components_gauss = n_components_gauss

        # spectral clustering
        self.n_clusters_spectral = n_clusters_spectral

        # agglomerative
        self.n_clusters_agg = n_clusters_agg
        self.affinity_agg = affinity_agg
        self.linkage_agg = linkage_agg
        if distance_threshold == -1:
            self.distance_threshold = None
        else:
            self.distance_threshold = distance_threshold

        #  birch
        self.threshold_birch = threshold_birch
        self.branching_factor_birch = branching_factor_birch
        self.n_clusters_birch = n_clusters_birch

        # dbscan
        self.eps_dbscan = eps_dbscan
        self.min_samples_dbscan = min_samples_dbscan


# stores the parameters for the Feature Selection Algorithm
class InputDataFeatureSelection:

    def __init__(self, selection_algorithm='PCA', seed=0,
                 n_features_pca=0,
                 variance_var=0.8,
                 ignore_var='NONE',
                 n_components_sparse=-1,
                 n_components_gaussian=-1):

        self.selection_algorithm = selection_algorithm
        self.seed = seed

        # pca
        self.n_features_pca = n_features_pca

        # variance
        self.variance_var = variance_var
        self.ignore_var = ignore_var


        # sparse
        if n_components_sparse == -1:
            self.n_components_sparse = 'auto'
        else:
            self.n_components_sparse = n_components_sparse

        # gaussian
        if n_components_gaussian == -1:
            self.n_components_gaussian = 'auto'
        else:
            self.n_components_gaussian = n_components_gaussian


# Stores the parameters for the Scaling Algorithm
class InputDataScaling:

    def __init__(self, scaling_algorithm='SCALEMINUSPLUS1', scaling_technique='NORMALSCALE', scaling_k_best=3):
        self.scaling_algorithm = scaling_algorithm
        self.scaling_technique = scaling_technique
        self.scaling_k_best = scaling_k_best
