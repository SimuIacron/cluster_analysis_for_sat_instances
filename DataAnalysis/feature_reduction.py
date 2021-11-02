from sklearn.decomposition import PCA
from sklearn.feature_selection import VarianceThreshold, SelectPercentile, chi2
from sklearn.random_projection import GaussianRandomProjection, SparseRandomProjection

from DataFormats.InputData import InputDataFeatureSelection

FEATURESELECTIONALGORITHMS = [
    # ('sparse', 'SPARSE'), # currently broken
    # ('gaussian', 'GAUSSIAN'), # currently broken
    ('Variance', 'VARIANCE'),
    ('PCA', 'PCA'),
    ('No feature selection', 'NONE')
]


def feature_reduction(data, params: InputDataFeatureSelection):
    algorithm = params.selection_algorithm

    if algorithm == "SPARSE":
        model = SparseRandomProjection(n_components=params.n_features)
    elif algorithm == "GAUSSIAN":
        model = GaussianRandomProjection(n_components=params.n_features)
    elif algorithm == "NONE":
        return data
    elif algorithm == "VARIANCE":
        sel = VarianceThreshold(threshold=(params.variance * (1 - params.variance)))
        return sel.fit_transform(data)
    else:  # algorithm == "PCA
        if params.n_features == 0:
            model = PCA(n_components="mle", svd_solver="full")
        else:
            model = PCA(n_components=params.n_features)

    model.fit(data)
    return model.transform(data)
