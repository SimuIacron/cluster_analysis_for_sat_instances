from sklearn.decomposition import PCA
from sklearn.feature_selection import VarianceThreshold, SelectPercentile, chi2
from sklearn.random_projection import GaussianRandomProjection, SparseRandomProjection

import DatabaseReader
import util
from DataFormats.InputData import InputDataFeatureSelection
import numpy as np

FEATURESELECTIONALGORITHMS = [
    ('Variance', 'VARIANCE'),
    ('PCA', 'PCA'),
    ('sparse', 'SPARSE'),  # currently broken
    ('gaussian', 'GAUSSIAN'),  # currently broken
    ('No feature selection', 'NONE')
]

FEATURESELECTIONIGNORE = [
    ('Variance applies to all features ', 'NONE'),
    ('Variance ignores solver time', 'TIMEIGNORE')
]


def feature_reduction(data, features, params: InputDataFeatureSelection):
    algorithm = params.selection_algorithm

    ignore_solver_time = True

    if algorithm == "SPARSE":
        model = SparseRandomProjection(random_state=params.seed, n_components=params.n_components_sparse)
    elif algorithm == "GAUSSIAN":
        model = GaussianRandomProjection(random_state=params.seed, n_components=params.n_components_gaussian)
    elif algorithm == "NONE":
        return data
    elif algorithm == "VARIANCE":


        if ignore_solver_time:

            # this section only calculates the variance on the instance features (e.g. base and gate)
            # but ignores the solver times, otherwise many of the solver times get removed
            # enabling this will keep all features of solver times
            # IMPORTANT: It is assumed that the solver time features are the last in the list
            solver_start_index = -1
            if DatabaseReader.FEATURES_SOLVER[0] in features:
                solver_start_index = features.index(DatabaseReader.FEATURES_SOLVER[0])

            calc_data = data
            # make sure that there are solver time features in the data
            if solver_start_index != -1:
                calc_data = [item[:solver_start_index] for item in data]

            sel = VarianceThreshold(threshold=(params.variance_var * (1 - params.variance_var)))

            reduced = sel.fit_transform(calc_data)
            finished = reduced
            if solver_start_index != -1:
                finished = []
                for i in range(len(reduced)):
                    entry = list(reduced[i]) + list(data[i][solver_start_index:])
                    finished.append(entry)
                print('Remaining features:')
                print(list(sel.get_feature_names_out(features[:solver_start_index])) + list(features[solver_start_index:]))
            else:
                print('Remaining features:')
                print(sel.get_feature_names_out(features))
        else:

            # this section calculates the variance for all features including the solver time

            sel = VarianceThreshold(threshold=(params.variance_var * (1 - params.variance_var)))
            finished = sel.fit_transform(data)
            print('Remaining features:')
            print(sel.get_feature_names_out(features))

        print('Remaining number of features: ' + str(len(finished[0])))
        return finished

    else:  # algorithm == "PCA
        if params.n_features_pca == 0:
            model = PCA(n_components="mle", svd_solver="full")
        else:
            model = PCA(n_components=params.n_features_pca)

    model.fit(data)
    return model.transform(data)
