from sklearn.decomposition import PCA
from sklearn.feature_selection import VarianceThreshold, SelectPercentile, mutual_info_classif, SelectKBest
from sklearn.random_projection import GaussianRandomProjection, SparseRandomProjection
from DataFormats import DatabaseReader
from numpy import argmin

FEATURESELECTIONALGORITHMS = [
    ('Mutual info', 'MUTUALINFO'),
    ('No feature selection', 'NONE'),
    ('Variance', 'VARIANCE'),
    ('PCA', 'PCA'),
    ('sparse', 'SPARSE'),
    ('gaussian', 'GAUSSIAN')
]

FEATURESELECTIONIGNORE = [
    ('Variance applies to all features ', 'NONE'),
    ('Variance ignores solver time', 'TIMEIGNORE')
]


def get_best_solver_per_instance(solvers):
    best_list = []
    for inst in solvers:
        index = argmin(inst)
        best_list.append(index)

    return best_list


def feature_selection(data, features, solvers, params_dict):
    algorithm = params_dict['selection_algorithm']

    ignore_solver_time = True

    if algorithm == "SPARSE":
        model = SparseRandomProjection(random_state=params_dict['seed'], n_components=params_dict['n_components_sparse'])
    elif algorithm == "MUTUALINFO":

        best_solver = get_best_solver_per_instance(solvers)
        sel = SelectPercentile(mutual_info_classif, percentile=params_dict['percentile_best'])
        reduced = sel.fit_transform(data, best_solver)

        print("Remaining features: " + str(len(reduced[0])))
        print("Remaining features:")
        print(sel.get_feature_names_out(features))
        return reduced

    elif algorithm == "MUTUALINFOK":
        best_solver = get_best_solver_per_instance(solvers)
        sel = SelectKBest(mutual_info_classif, k=params_dict['percentile_best'])
        reduced = sel.fit_transform(data, best_solver)

        print("Remaining features: " + str(len(reduced[0])))
        print("Remaining features:")
        print(sel.get_feature_names_out(features))
        return reduced

    elif algorithm == "GAUSSIAN":
        model = GaussianRandomProjection(random_state=params_dict['seed'], n_components=params_dict['n_components_gaussian'])
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

            sel = VarianceThreshold(threshold=(params_dict['variance_var'] * (1 - params_dict['variance_var'])))

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

            sel = VarianceThreshold(threshold=(params_dict['variance_var'] * (1 - params_dict['variance_var'])))
            finished = sel.fit_transform(data)
            print('Remaining features:')
            print(sel.get_feature_names_out(features))

        print('Remaining number of features: ' + str(len(finished[0])))
        return finished

    else:  # algorithm == "PCA
        if params_dict['n_features_pca'] == 0:
            model = PCA(n_components="mle", svd_solver="full")
        else:
            model = PCA(n_components=params_dict['n_features_pca'])

    model.fit(data)
    return model.transform(data)
