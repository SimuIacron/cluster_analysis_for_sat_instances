from sklearn.decomposition import PCA
from sklearn.feature_selection import VarianceThreshold, SelectPercentile, chi2
from sklearn.random_projection import GaussianRandomProjection, SparseRandomProjection

from DataFormats.InputData import InputDataFeatureSelection

FEATURESELECTIONALGORITHMS = [
    ('Variance', 'VARIANCE'),
    ('PCA', 'PCA'),
    ('sparse', 'SPARSE'),  # currently broken
    ('gaussian', 'GAUSSIAN'),  # currently broken
    ('No feature selection', 'NONE')
]


def feature_reduction(data, params: InputDataFeatureSelection):
    algorithm = params.selection_algorithm

    if algorithm == "SPARSE":
        model = SparseRandomProjection(random_state=params.seed, n_components=params.n_components_sparse)
    elif algorithm == "GAUSSIAN":
        model = GaussianRandomProjection(random_state=params.seed, n_components=params.n_components_gaussian)
    elif algorithm == "NONE":
        return data
    elif algorithm == "VARIANCE":
        sel = VarianceThreshold(threshold=(params.variance_var * (1 - params.variance_var)))
        return sel.fit_transform(data)
    else:  # algorithm == "PCA
        if params.n_features_pca == 0:
            model = PCA(n_components="mle", svd_solver="full")
        else:
            model = PCA(n_components=params.n_features_pca)

    model.fit(data)
    return model.transform(data)
